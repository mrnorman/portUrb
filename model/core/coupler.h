
#pragma once

#include "main_header.h"
#include "DataManager.h"
#include "YAKL_pnetcdf.h"
#include "YAKL_netcdf.h"
#include "MultipleFields.h"
#include "Options.h"

// The Coupler class holds everything a component or module of this model would need in order to perform its
// changes to the model state


namespace core {

  class Coupler {
  protected:
    Options options;

    real xlen;   // Domain length in the x-direction in meters
    real ylen;   // Domain length in the y-direction in meters
    real zlen;   // Domain length in the z-direction in meters
    real dt_gcm; // Time step of the GCM for this MMF invocation

    // MPI parallelization information
    int    nranks;           // Total number of MPI ranks / processes
    int    myrank;           // My rank # in [0,nranks-1]
    int    nens;             // Number of ensembles
    size_t nx_glob;          // Total global number of cells in the x-direction (summing all MPI Processes)
    size_t ny_glob;          // Total global number of cells in the y-direction (summing all MPI Processes)
    int    nproc_x;          // Number of parallel processes distributed over the x-dimension
    int    nproc_y;          // Number of parallel processes distributed over the y-dimension
                             // nproc_x * nproc_y  must equal  nranks
    int    px;               // My process ID in the x-direction
    int    py;               // My process ID in the y-direction
    size_t i_beg;            // Beginning of my x-direction global index
    size_t j_beg;            // Beginning of my y-direction global index
    size_t i_end;            // End of my x-direction global index
    size_t j_end;            // End of my y-direction global index
    bool   mainproc;         // myrank == 0
    SArray<int,2,3,3> neigh; // List of neighboring rank IDs;  1st index: y;  2nd index: x
                             // Y: 0 = south;  1 = middle;  2 = north
                             // X: 0 = west ;  1 = center;  3 = east 
    int    file_counter;

    DataManager dm;

    struct Tracer {
      std::string name;
      std::string desc;
      bool        positive;
      bool        adds_mass;
      bool        diffuse;
    };
    std::vector<Tracer> tracers;

    struct OutputVar {
      std::string name;
      int         dims;
      size_t      type_hash;
    };
    std::vector<OutputVar> output_vars;


  public:

    int static constexpr DIMS_COLUMN  = 1;
    int static constexpr DIMS_SURFACE = 2;
    int static constexpr DIMS_3D      = 3;

    Coupler() {
      this->xlen   = -1;
      this->ylen   = -1;
      this->zlen   = -1;
      this->dt_gcm = -1;
      this->file_counter = 0;
    }


    Coupler(Coupler &&) = default;
    Coupler &operator=(Coupler &&) = default;
    Coupler(Coupler const &) = delete;
    Coupler &operator=(Coupler const &) = delete;


    ~Coupler() {
      yakl::fence();
      dm.finalize();
      options.finalize();
      tracers = std::vector<Tracer>();
      this->xlen   = -1;
      this->ylen   = -1;
      this->zlen   = -1;
      this->dt_gcm = -1;
    }


    void clone_into( Coupler &coupler ) {
      coupler.xlen     = this->xlen    ;
      coupler.ylen     = this->ylen    ;
      coupler.zlen     = this->zlen    ;
      coupler.dt_gcm   = this->dt_gcm  ;
      coupler.tracers  = this->tracers ;
      coupler.nranks   = this->nranks  ;
      coupler.myrank   = this->myrank  ;
      coupler.nens     = this->nens    ;
      coupler.nx_glob  = this->nx_glob ;
      coupler.ny_glob  = this->ny_glob ;
      coupler.nproc_x  = this->nproc_x ;
      coupler.nproc_y  = this->nproc_y ;
      coupler.px       = this->px      ;
      coupler.py       = this->py      ;
      coupler.i_beg    = this->i_beg   ;
      coupler.j_beg    = this->j_beg   ;
      coupler.i_end    = this->i_end   ;
      coupler.j_end    = this->j_end   ;
      coupler.mainproc = this->mainproc;
      coupler.neigh    = this->neigh   ;
      this->dm.clone_into( coupler.dm );
    }


    void distribute_mpi_and_allocate_coupled_state(int nz, size_t ny_glob, size_t nx_glob, int nens,
                                                   int nproc_x_in = -1 , int nproc_y_in = -1 ,
                                                   int px_in      = -1 , int py_in      = -1 ,
                                                   int i_beg_in   = -1 , int i_end_in   = -1 ,
                                                   int j_beg_in   = -1 , int j_end_in   = -1 ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      this->nens    = nens   ;
      this->nx_glob = nx_glob;
      this->ny_glob = ny_glob;

      MPI_Comm_size( MPI_COMM_WORLD , &nranks );
      MPI_Comm_rank( MPI_COMM_WORLD , &myrank );

      mainproc = (myrank == 0);

      bool sim2d = ny_glob == 1;

      if (sim2d) {
        nproc_x = nranks;
        nproc_y = 1;
      } else {
        // Find integer nproc_y * nproc_x == nranks such that nproc_y and nproc_x are as close as possible
        nproc_y = (int) std::ceil( std::sqrt((double) nranks) );
        while (nproc_y >= 1) {
          if (nranks % nproc_y == 0) { break; }
          nproc_y--;
        }
        nproc_x = nranks / nproc_y;
      }

      // Get my ID within each dimension's number of ranks
      py = myrank / nproc_x;
      px = myrank % nproc_x;

      // Get my beginning and ending indices in the x- and y- directions
      double nper;
      nper = ((double) nx_glob)/nproc_x;
      i_beg = static_cast<size_t>( round( nper* px    )   );
      i_end = static_cast<size_t>( round( nper*(px+1) )-1 );
      nper = ((double) ny_glob)/nproc_y;
      j_beg = static_cast<size_t>( round( nper* py    )   );
      j_end = static_cast<size_t>( round( nper*(py+1) )-1 );

      // For multi-resolution experiments, the user might want to set these manually to ensure that
      //   grids match up properly when decomposed into ranks
      if (nproc_x_in > 0) nproc_x = nproc_x_in;
      if (nproc_y_in > 0) nproc_y = nproc_y_in;
      if (px_in      > 0) px      = px_in     ;
      if (py_in      > 0) py      = py_in     ;
      if (i_beg_in   > 0) i_beg   = i_beg_in  ;
      if (i_end_in   > 0) i_end   = i_end_in  ;
      if (j_beg_in   > 0) j_beg   = j_beg_in  ;
      if (j_end_in   > 0) j_end   = j_end_in  ;

      //Determine my number of grid cells
      int nx = i_end - i_beg + 1;
      int ny = j_end - j_beg + 1;
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
          int pxloc = px+i-1;
          while (pxloc < 0        ) { pxloc = pxloc + nproc_x; }
          while (pxloc > nproc_x-1) { pxloc = pxloc - nproc_x; }
          int pyloc = py+j-1;
          while (pyloc < 0        ) { pyloc = pyloc + nproc_y; }
          while (pyloc > nproc_y-1) { pyloc = pyloc - nproc_y; }
          neigh(j,i) = pyloc * nproc_x + pxloc;
        }
      }

      // Debug output for the parallel decomposition
      #if 0
        if (mainproc) {
          std::cout << "There are " << nranks << " ranks, with " << nproc_x << " in the x-direction and "
                                                                 << nproc_y << " in the y-direction.\n\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (int rr=0; rr < nranks; rr++) {
          MPI_Barrier(MPI_COMM_WORLD);
          if (rr == myrank) {
            std::cout << "Hello! My Rank is    : " << myrank << std::endl;
            std::cout << "My proc grid ID is   : " << px << " , " << py << std::endl;
            std::cout << "I have               : " << nx << " x " << ny << " x " << nz <<  " columns." << std::endl;
            std::cout << "I start at index     : " << i_beg << " x " << j_beg << std::endl;
            std::cout << "I end at index       : " << i_end << " x " << j_end << std::endl;
            std::cout << "My neighbor matrix is:" << std::endl;
            for (int j = 2; j >= 0; j--) {
              for (int i = 0; i < 3; i++) {
                std::cout << std::setw(6) << neigh(j,i) << " ";
              }
              std::cout << std::endl;
            }
            std::cout << std::endl;
          }
          MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      #endif

      dm.add_dimension( "nens" , nens );
      dm.add_dimension( "x"    , nx   );
      dm.add_dimension( "y"    , ny   );
      dm.add_dimension( "z"    , nz   );
      set_option<real>("elapsed_time",0._fp);
    }


    void set_dt_gcm(real dt_gcm) { this->dt_gcm = dt_gcm; }

    real                      get_xlen                  () const { return this->xlen        ; }
    real                      get_ylen                  () const { return this->ylen        ; }
    real                      get_zlen                  () const { return this->zlen        ; }
    real                      get_dt_gcm                () const { return this->dt_gcm      ; }
    int                       get_nranks                () const { return this->nranks      ; }
    int                       get_myrank                () const { return this->myrank      ; }
    int                       get_nens                  () const { return this->nens        ; }
    size_t                    get_nx_glob               () const { return this->nx_glob     ; }
    size_t                    get_ny_glob               () const { return this->ny_glob     ; }
    int                       get_nproc_x               () const { return this->nproc_x     ; }
    int                       get_nproc_y               () const { return this->nproc_y     ; }
    int                       get_px                    () const { return this->px          ; }
    int                       get_py                    () const { return this->py          ; }
    size_t                    get_i_beg                 () const { return this->i_beg       ; }
    size_t                    get_j_beg                 () const { return this->j_beg       ; }
    size_t                    get_i_end                 () const { return this->i_end       ; }
    size_t                    get_j_end                 () const { return this->j_end       ; }
    bool                      is_sim2d                  () const { return this->ny_glob == 1; }
    bool                      is_mainproc               () const { return this->mainproc    ; }
    SArray<int,2,3,3> const & get_neighbor_rankid_matrix() const { return this->neigh       ; }
    DataManager       const & get_data_manager_readonly () const { return this->dm          ; }
    DataManager             & get_data_manager_readwrite()       { return this->dm          ; }


    int get_nx() const {
      if (dm.find_dimension("x") == -1) return -1;
      return dm.get_dimension_size("x");
    }


    int get_ny() const {
      if (dm.find_dimension("y") == -1) return -1;
      return dm.get_dimension_size("y");
    }


    int get_nz() const {
      if (dm.find_dimension("z") == -1) return -1;
      return dm.get_dimension_size("z");
    }


    real get_dx() const { return get_xlen() / nx_glob; }


    real get_dy() const { return get_ylen() / ny_glob; }


    real get_dz() const { return get_zlen() / get_nz(); }


    int get_num_tracers() const { return tracers.size(); }


    MPI_Datatype get_mpi_data_type() const {
      if      constexpr (std::is_same<real,float >()) { return MPI_FLOAT; }
      else if constexpr (std::is_same<real,double>()) { return MPI_DOUBLE; }
      else { endrun("ERROR: Invalid type for 'real'"); }
      return MPI_FLOAT;
    }


    template <class T>
    void add_option( std::string key , T value ) {
      options.add_option<T>(key,value);
    }


    template <class T>
    void set_option( std::string key , T value ) {
      options.set_option<T>(key,value);
    }


    template <class T>
    T get_option( std::string key ) const {
      return options.get_option<T>(key);
    }


    template <class T>
    T get_option( std::string key , T val ) const {
      if (option_exists(key)) return options.get_option<T>(key);
      return val;
    }


    bool option_exists( std::string key ) const {
      return options.option_exists(key);
    }


    void delete_option( std::string key ) {
      options.delete_option(key);
    }


    void set_grid(real xlen, real ylen, real zlen) {
      this->xlen = xlen;
      this->ylen = ylen;
      this->zlen = zlen;
    }

    
    int add_tracer( std::string tracer_name      ,
                    std::string tracer_desc = "" ,
                    bool positive  = true        ,
                    bool adds_mass = false       ,
                    bool diffuse   = true        ) {
      int nz   = get_nz();
      int ny   = get_ny();
      int nx   = get_nx();
      int nens = get_nens();
      dm.register_and_allocate<real>( tracer_name , tracer_desc , {nz,ny,nx,nens} , {"z","y","x","nens"} );
      tracers.push_back( { tracer_name , tracer_desc , positive , adds_mass , diffuse } );
      return tracers.size()-1;
    }

    
    std::vector<std::string> get_tracer_names() const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) { ret.push_back( tracers[i].name ); }
      return ret;
    }

    
    void get_tracer_info(std::string tracer_name , std::string &tracer_desc, bool &tracer_found ,
                         bool &positive , bool &adds_mass, bool &diffuse) const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) {
        if (tracer_name == tracers[i].name) {
          positive    = tracers[i].positive ;
          tracer_desc = tracers[i].desc     ;
          adds_mass   = tracers[i].adds_mass;
          diffuse     = tracers[i].diffuse  ;
          tracer_found = true;
          return;
        }
      }
      tracer_found = false;
    }

    
    bool tracer_exists( std::string tracer_name ) const {
      for (int i=0; i < tracers.size(); i++) {
        if (tracer_name == tracers[i].name) return true;
      }
      return false;
    }


    template <class T> size_t get_type_hash() const {
      return typeid(typename std::remove_cv<T>::type).hash_code();
    }


    template <class T> void register_output_variable( std::string name , int dims ) {
      output_vars.push_back({name,dims,get_type_hash<T>()});
    }


    void write_output_file( std::string prefix ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      typedef unsigned char uchar;
      yakl::timer_start("output");
      auto nens        = get_nens();
      auto nx          = get_nx();
      auto ny          = get_ny();
      auto nz          = get_nz();
      auto dx          = get_dx();
      auto dy          = get_dy();
      auto dz          = get_dz();
      auto num_tracers = get_num_tracers();
      auto C0          = get_option<real>("C0");
      auto R_d         = get_option<real>("R_d");
      auto gamma       = get_option<real>("gamma_d");
      auto etime       = get_option<real>("elapsed_time");
      int i_beg        = get_i_beg();
      int j_beg        = get_j_beg();
      yakl::SimplePNetCDF nc;
      std::stringstream fname;
      fname << prefix << "_" << std::setw(8) << std::setfill('0') << file_counter << ".nc";
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info, "romio_no_indep_rw",    "true");
      MPI_Info_set(info, "nc_header_align_size", "1048576");
      MPI_Info_set(info, "nc_var_align_size",    "1048576");
      nc.create(fname.str() , NC_CLOBBER | NC_64BIT_DATA , info );
      nc.create_dim( "ens" , nens );
      nc.create_dim( "x"   , get_nx_glob() );
      nc.create_dim( "y"   , get_ny_glob() );
      nc.create_dim( "z"   , nz );
      nc.create_dim( "t"   , 1 );
      std::vector<std::string> dimnames_column  = {"z","ens"};
      std::vector<std::string> dimnames_surface = {"y","x","ens"};
      std::vector<std::string> dimnames_3d      = {"z","y","x","ens"};
      nc.create_var<real>( "x"   , {"x"} );
      nc.create_var<real>( "y"   , {"y"} );
      nc.create_var<real>( "z"   , {"z"} );
      nc.create_var<real>( "density_dry"  , dimnames_3d );
      nc.create_var<real>( "uvel"         , dimnames_3d );
      nc.create_var<real>( "vvel"         , dimnames_3d );
      nc.create_var<real>( "wvel"         , dimnames_3d );
      nc.create_var<real>( "temperature"  , dimnames_3d );
      nc.create_var<real>( "etime"        , {"t"} );
      nc.create_var<real>( "file_counter" , {"t"} );
      auto tracer_names = get_tracer_names();
      for (int tr = 0; tr < num_tracers; tr++) { nc.create_var<real>( tracer_names[tr] , dimnames_3d ); }
      for (int ivar = 0; ivar < output_vars.size(); ivar++) {
        auto name = output_vars[ivar].name;
        auto hash = output_vars[ivar].type_hash;
        auto dims = output_vars[ivar].dims;
        if        (dims == DIMS_COLUMN ) {
          if      (hash == get_type_hash<float >()) { nc.create_var<float >(name,dimnames_column ); }
          else if (hash == get_type_hash<double>()) { nc.create_var<double>(name,dimnames_column ); }
          else if (hash == get_type_hash<int   >()) { nc.create_var<int   >(name,dimnames_column ); }
          else if (hash == get_type_hash<uchar >()) { nc.create_var<uchar >(name,dimnames_column ); }
        } else if (dims == DIMS_SURFACE) {
          if      (hash == get_type_hash<float >()) { nc.create_var<float >(name,dimnames_surface); }
          else if (hash == get_type_hash<double>()) { nc.create_var<double>(name,dimnames_surface); }
          else if (hash == get_type_hash<int   >()) { nc.create_var<int   >(name,dimnames_surface); }
          else if (hash == get_type_hash<uchar >()) { nc.create_var<uchar >(name,dimnames_surface); }
        } else if (dims == DIMS_3D     ) {
          if      (hash == get_type_hash<float >()) { nc.create_var<float >(name,dimnames_3d     ); }
          else if (hash == get_type_hash<double>()) { nc.create_var<double>(name,dimnames_3d     ); }
          else if (hash == get_type_hash<int   >()) { nc.create_var<int   >(name,dimnames_3d     ); }
          else if (hash == get_type_hash<uchar >()) { nc.create_var<uchar >(name,dimnames_3d     ); }
        }
      }
      nc.enddef();
      // x-coordinate
      real1d xloc("xloc",nx);
      parallel_for( YAKL_AUTO_LABEL() , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
      nc.write_all( xloc , "x" , {i_beg} );
      // y-coordinate
      real1d yloc("yloc",ny);
      parallel_for( YAKL_AUTO_LABEL() , ny , YAKL_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
      nc.write_all( yloc , "y" , {j_beg} );
      // z-coordinate
      real1d zloc("zloc",nz);
      parallel_for( YAKL_AUTO_LABEL() , nz , YAKL_LAMBDA (int k) { zloc(k) = (k      +0.5)*dz; });
      nc.begin_indep_data();
      if (is_mainproc()) nc.write( zloc , "z" );
      if (is_mainproc()) nc.write( etime        , "etime"        );
      if (is_mainproc()) nc.write( file_counter , "file_counter" );
      nc.end_indep_data();
      auto &dm = get_data_manager_readonly();
      std::vector<MPI_Offset> start_3d      = {0,j_beg,i_beg,0};
      std::vector<MPI_Offset> start_surface = {  j_beg,i_beg,0};
      nc.write_all(dm.get<real const,4>("density_dry"),"density_dry",start_3d);
      nc.write_all(dm.get<real const,4>("uvel"       ),"uvel"       ,start_3d);
      nc.write_all(dm.get<real const,4>("vvel"       ),"vvel"       ,start_3d);
      nc.write_all(dm.get<real const,4>("wvel"       ),"wvel"       ,start_3d);
      nc.write_all(dm.get<real const,4>("temp"       ),"temperature",start_3d);
      for (int i=0; i < tracer_names.size(); i++) {
        nc.write_all(dm.get<real const,4>(tracer_names[i]),tracer_names[i],start_3d);
      }
      for (int ivar = 0; ivar < output_vars.size(); ivar++) {
        auto name = output_vars[ivar].name;
        auto hash = output_vars[ivar].type_hash;
        auto dims = output_vars[ivar].dims;
        if        (dims == DIMS_COLUMN ) {
          nc.begin_indep_data();
          if (is_mainproc()) {
            if      (hash == get_type_hash<float >()) { nc.write(dm.get<float  const,2>(name),name); }
            else if (hash == get_type_hash<double>()) { nc.write(dm.get<double const,2>(name),name); }
            else if (hash == get_type_hash<int   >()) { nc.write(dm.get<int    const,2>(name),name); }
            else if (hash == get_type_hash<uchar >()) { nc.write(dm.get<uchar  const,2>(name),name); }
          }
          nc.end_indep_data();
        } else if (dims == DIMS_SURFACE) {
          if      (hash == get_type_hash<float >()) { nc.write_all(dm.get<float  const,3>(name),name,start_surface); }
          else if (hash == get_type_hash<double>()) { nc.write_all(dm.get<double const,3>(name),name,start_surface); }
          else if (hash == get_type_hash<int   >()) { nc.write_all(dm.get<int    const,3>(name),name,start_surface); }
          else if (hash == get_type_hash<uchar >()) { nc.write_all(dm.get<uchar  const,3>(name),name,start_surface); }
        } else if (dims == DIMS_3D     ) {
          if      (hash == get_type_hash<float >()) { nc.write_all(dm.get<float  const,4>(name),name,start_3d); }
          else if (hash == get_type_hash<double>()) { nc.write_all(dm.get<double const,4>(name),name,start_3d); }
          else if (hash == get_type_hash<int   >()) { nc.write_all(dm.get<int    const,4>(name),name,start_3d); }
          else if (hash == get_type_hash<uchar >()) { nc.write_all(dm.get<uchar  const,4>(name),name,start_3d); }
        }
      }
      nc.close();
      file_counter++;
      yakl::timer_stop("output");
    }


    void overwrite_with_restart() {
      typedef unsigned char uchar;
      yakl::timer_start("overwrite_with_restart");
      if (is_mainproc())  std::cout << "*** Restarting from file: "
                                    << get_option<std::string>("restart_file") << std::endl;
      int i_beg = get_i_beg();
      int j_beg = get_j_beg();
      auto tracer_names = get_tracer_names();
      yakl::SimplePNetCDF nc;
      nc.open( get_option<std::string>("restart_file") , NC_NOWRITE );
      nc.begin_indep_data();
      real etime;
      if (is_mainproc()) nc.read( etime        , "etime"        );
      if (is_mainproc()) nc.read( file_counter , "file_counter" );
      set_option<real>("elapsed_time",etime);
      nc.end_indep_data();
      MPI_Bcast( &file_counter , 1 , MPI_INT , 0 , MPI_COMM_WORLD );
      std::vector<MPI_Offset> start_3d      = {0,j_beg,i_beg,0};
      std::vector<MPI_Offset> start_surface = {  j_beg,i_beg,0};
      std::vector<MPI_Offset> start_column  = {0            ,0};
      nc.read_all(dm.get<real,4>("density_dry"),"density_dry",start_3d);
      nc.read_all(dm.get<real,4>("uvel"       ),"uvel"       ,start_3d);
      nc.read_all(dm.get<real,4>("vvel"       ),"vvel"       ,start_3d);
      nc.read_all(dm.get<real,4>("wvel"       ),"wvel"       ,start_3d);
      nc.read_all(dm.get<real,4>("temp"       ),"temperature",start_3d);
      for (int i=0; i < tracer_names.size(); i++) {
        nc.read_all(dm.get<real,4>(tracer_names[i]),tracer_names[i],start_3d);
      }
      for (int ivar = 0; ivar < output_vars.size(); ivar++) {
        auto name = output_vars[ivar].name;
        auto hash = output_vars[ivar].type_hash;
        auto dims = output_vars[ivar].dims;
        if        (dims == DIMS_COLUMN ) {
          if      (hash == get_type_hash<float >()) { nc.read_all(dm.get<float ,2>(name),name,start_column); }
          else if (hash == get_type_hash<double>()) { nc.read_all(dm.get<double,2>(name),name,start_column); }
          else if (hash == get_type_hash<int   >()) { nc.read_all(dm.get<int   ,2>(name),name,start_column); }
          else if (hash == get_type_hash<uchar >()) { nc.read_all(dm.get<uchar ,2>(name),name,start_column); }
        } else if (dims == DIMS_SURFACE) {
          if      (hash == get_type_hash<float >()) { nc.read_all(dm.get<float ,3>(name),name,start_surface); }
          else if (hash == get_type_hash<double>()) { nc.read_all(dm.get<double,3>(name),name,start_surface); }
          else if (hash == get_type_hash<int   >()) { nc.read_all(dm.get<int   ,3>(name),name,start_surface); }
          else if (hash == get_type_hash<uchar >()) { nc.read_all(dm.get<uchar ,3>(name),name,start_surface); }
        } else if (dims == DIMS_3D     ) {
          if      (hash == get_type_hash<float >()) { nc.read_all(dm.get<float ,4>(name),name,start_3d); }
          else if (hash == get_type_hash<double>()) { nc.read_all(dm.get<double,4>(name),name,start_3d); }
          else if (hash == get_type_hash<int   >()) { nc.read_all(dm.get<int   ,4>(name),name,start_3d); }
          else if (hash == get_type_hash<uchar >()) { nc.read_all(dm.get<uchar ,4>(name),name,start_3d); }
        }
      }
      nc.close();
      file_counter++;
      yakl::timer_stop("overwrite_with_restart");
    }

  };

}


