
#include "dynamics_rk.h"

namespace modules {

  void Dynamics_Euler_Stratified_WenoFV::init(core::Coupler &coupler) const {
    #ifdef YAKL_AUTO_PROFILE
      MPI_Barrier(MPI_COMM_WORLD);
      yakl::timer_start("init");
    #endif
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nx             = coupler.get_nx();
    auto ny             = coupler.get_ny();
    auto nz             = coupler.get_nz();
    auto dz             = coupler.get_dz();
    auto nx_glob        = coupler.get_nx_glob();
    auto ny_glob        = coupler.get_ny_glob();
    auto num_tracers    = coupler.get_num_tracers();
    auto gamma          = coupler.get_option<real>("gamma_d");
    auto C0             = coupler.get_option<real>("C0"     );
    auto grav           = coupler.get_option<real>("grav"   );
    auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);

    num_tracers = coupler.get_num_tracers();
    bool1d tracer_adds_mass("tracer_adds_mass",num_tracers);
    bool1d tracer_positive ("tracer_positive" ,num_tracers);
    auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
    auto tracer_positive_host  = tracer_positive .createHostCopy();
    auto tracer_names = coupler.get_tracer_names();
    for (int tr=0; tr < num_tracers; tr++) {
      std::string tracer_desc;
      bool        tracer_found, positive, adds_mass, diffuse;
      coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass , diffuse );
      tracer_positive_host (tr) = positive;
      tracer_adds_mass_host(tr) = adds_mass;
    }
    tracer_positive_host .deep_copy_to(tracer_positive );
    tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);
    auto &dm = coupler.get_data_manager_readwrite();
    dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
    auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
    tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);
    dm.register_and_allocate<bool>("tracer_positive","",{num_tracers});
    auto dm_tracer_positive = dm.get<bool,1>("tracer_positive");
    tracer_positive.deep_copy_to(dm_tracer_positive);

    real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);  state   = 0;
    real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);  tracers = 0;
    convert_coupler_to_dynamics( coupler , state , tracers );
    dm.register_and_allocate<real>("hy_dens_cells"    ,"",{nz+2*hs});
    dm.register_and_allocate<real>("hy_theta_cells"   ,"",{nz+2*hs});
    dm.register_and_allocate<real>("hy_pressure_cells","",{nz+2*hs});
    auto r = dm.get<real,1>("hy_dens_cells"    );    r = 0;
    auto t = dm.get<real,1>("hy_theta_cells"   );    t = 0;
    auto p = dm.get<real,1>("hy_pressure_cells");    p = 0;
    if (enable_gravity) {
      parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , YAKL_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r(k) += state(idR,k,hs+j,hs+i);
            t(k) += state(idT,k,hs+j,hs+i) / state(idR,k,hs+j,hs+i);
            p(k) += C0 * std::pow( state(idT,k,hs+j,hs+i) , gamma );
          }
        }
      });
      auto r_loc = r .createHostCopy();    auto r_glob = r .createHostObject();
      auto t_loc = t .createHostCopy();    auto t_glob = t .createHostObject();
      auto p_loc = p .createHostCopy();    auto p_glob = p .createHostObject();
      auto dtype = coupler.get_mpi_data_type();
      MPI_Allreduce( r_loc.data() , r_glob.data() , r.size() , dtype , MPI_SUM , MPI_COMM_WORLD );
      MPI_Allreduce( t_loc.data() , t_glob.data() , t.size() , dtype , MPI_SUM , MPI_COMM_WORLD );
      MPI_Allreduce( p_loc.data() , p_glob.data() , p.size() , dtype , MPI_SUM , MPI_COMM_WORLD );
      r_glob.deep_copy_to(r);
      t_glob.deep_copy_to(t);
      p_glob.deep_copy_to(p);
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , YAKL_LAMBDA (int k) {
        r(k) *= r_nx_ny;
        t(k) *= r_nx_ny;
        p(k) *= r_nx_ny;
      });
      parallel_for( YAKL_AUTO_LABEL() , hs , YAKL_LAMBDA (int kk) {
        {
          int  k0       = hs;
          int  k        = k0-1-kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 + grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k) = theta0;
          p(k) = C0*std::pow(r(k)*theta0,gamma);
        }
        {
          int  k0       = hs+nz-1;
          int  k        = k0+1+kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 - grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k) = theta0;
          p(k) = C0*std::pow(r(k)*theta0,gamma);
        }
      });
    }

    auto create_immersed_proportion_halos = [] (core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;;
      auto nz     = coupler.get_nz  ();
      auto ny     = coupler.get_ny  ();
      auto nx     = coupler.get_nx  ();
      auto bc_z   = coupler.get_option<std::string>("bc_z","solid_wall");
      auto &dm    = coupler.get_data_manager_readwrite();
      if (!dm.entry_exists("dycore_immersed_proportion_halos")) {
        auto immersed_prop = dm.get<real,3>("immersed_proportion");
        core::MultiField<real,3> fields;
        fields.add_field( immersed_prop  );
        auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
        dm.register_and_allocate<real>("dycore_immersed_proportion_halos","",{nz+2*hs,ny+2*hs,nx+2*hs},
                                       {"z_halod","y_halod","x_halod"});
        dm.register_and_allocate<bool>("dycore_any_immersed","",{nz,ny,nx},
                                       {"z","y","x"});
        auto immersed_proportion_halos = dm.get<real,3>("dycore_immersed_proportion_halos");
        auto any_immersed = dm.get<bool,3>("dycore_any_immersed");
        fields_halos.get_field(0).deep_copy_to( immersed_proportion_halos );
        if (bc_z == "solid_wall") {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) , YAKL_LAMBDA (int kk, int j, int i) {
            immersed_proportion_halos(      kk,j,i) = 1;
            immersed_proportion_halos(hs+nz+kk,j,i) = 1;
          });
        }
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          any_immersed(k,j,i) = false;
          for (int kk=0; kk < ord; kk++) {
            for (int jj=0; jj < ord; jj++) {
              for (int ii=0; ii < ord; ii++) {
                if (immersed_proportion_halos(k+kk,j+jj,i+ii)) any_immersed(k,j,i) = true;
              }
            }
          }
        });
      }
    };

    auto compute_hydrostasis_edges = [] (core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;;
      auto nz   = coupler.get_nz  ();
      auto ny   = coupler.get_ny  ();
      auto nx   = coupler.get_nx  ();
      auto &dm  = coupler.get_data_manager_readwrite();
      if (! dm.entry_exists("hy_dens_edges" )) dm.register_and_allocate<real>("hy_dens_edges" ,"",{nz+1});
      if (! dm.entry_exists("hy_theta_edges")) dm.register_and_allocate<real>("hy_theta_edges","",{nz+1});
      auto hy_dens_cells  = dm.get<real const,1>("hy_dens_cells" );
      auto hy_theta_cells = dm.get<real const,1>("hy_theta_cells");
      auto hy_dens_edges  = dm.get<real      ,1>("hy_dens_edges" );
      auto hy_theta_edges = dm.get<real      ,1>("hy_theta_edges");
      if (ord < 5) {
        parallel_for( YAKL_AUTO_LABEL() , nz+1 , YAKL_LAMBDA (int k) {
          hy_dens_edges(k) = std::exp( 0.5_fp*std::log(hy_dens_cells(hs+k-1)) +
                                       0.5_fp*std::log(hy_dens_cells(hs+k  )) );
          hy_theta_edges(k) = 0.5_fp*hy_theta_cells(hs+k-1) +
                              0.5_fp*hy_theta_cells(hs+k  ) ;
        });
      } else {
        parallel_for( YAKL_AUTO_LABEL() , nz+1 , YAKL_LAMBDA (int k) {
          hy_dens_edges(k) = std::exp( -1./12.*std::log(hy_dens_cells(hs+k-2)) +
                                        7./12.*std::log(hy_dens_cells(hs+k-1)) +
                                        7./12.*std::log(hy_dens_cells(hs+k  )) +
                                       -1./12.*std::log(hy_dens_cells(hs+k+1)) );
          hy_theta_edges(k) = -1./12.*hy_theta_cells(hs+k-2) +
                               7./12.*hy_theta_cells(hs+k-1) +
                               7./12.*hy_theta_cells(hs+k  ) +
                              -1./12.*hy_theta_cells(hs+k+1);
        });
      }
    };

    create_immersed_proportion_halos( coupler );
    compute_hydrostasis_edges       ( coupler );

    // These are needed for a proper restart
    coupler.register_output_variable<real>( "immersed_proportion" , core::Coupler::DIMS_3D      );
    coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
      auto i_beg = coupler.get_i_beg();
      auto j_beg = coupler.get_j_beg();
      auto nz    = coupler.get_nz();
      auto ny    = coupler.get_ny();
      auto nx    = coupler.get_nx();
      nc.redef();
      nc.create_dim( "z_halo" , coupler.get_nz()+2*hs );
      nc.create_var<real>( "hy_dens_cells"     , {"z_halo"});
      nc.create_var<real>( "hy_theta_cells"    , {"z_halo"});
      nc.create_var<real>( "hy_pressure_cells" , {"z_halo"});
      nc.create_var<real>( "theta"             , {"z","y","x"});
      nc.enddef();
      nc.begin_indep_data();
      auto &dm = coupler.get_data_manager_readonly();
      if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_dens_cells"    ) , "hy_dens_cells"     );
      if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_theta_cells"   ) , "hy_theta_cells"    );
      if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_pressure_cells") , "hy_pressure_cells" );
      nc.end_indep_data();
      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      convert_coupler_to_dynamics( coupler , state , tracers );
      std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
      using yakl::componentwise::operator/;
      real3d data("data",nz,ny,nx);
      yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        data(k,j,i) = state(idT,hs+k,hs+j,hs+i) / state(idR,hs+k,hs+j,hs+i);
      });
      nc.write_all(data,"theta",start_3d);
    } );
    coupler.register_overwrite_with_restart_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
      auto &dm = coupler.get_data_manager_readwrite();
      nc.read_all(dm.get<real,1>("hy_dens_cells"    ),"hy_dens_cells"    ,{0});
      nc.read_all(dm.get<real,1>("hy_theta_cells"   ),"hy_theta_cells"   ,{0});
      nc.read_all(dm.get<real,1>("hy_pressure_cells"),"hy_pressure_cells",{0});
      create_immersed_proportion_halos( coupler );
      compute_hydrostasis_edges       ( coupler );
    } );
    #ifdef YAKL_AUTO_PROFILE
      yakl::timer_stop("init");
    #endif
  }

}


