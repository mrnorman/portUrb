
#include "sponge_layer.h"

namespace modules {

  void sponge_layer( core::Coupler &coupler , real dt , real time_scale , int num_layers ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;

    num_layers = std::max(2,num_layers);

    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto nz      = coupler.get_nz  ();
    auto ny      = coupler.get_ny  ();
    auto nx      = coupler.get_nx  ();
    auto zlen    = coupler.get_zlen();
    auto dz      = coupler.get_dz  ();

    int WFLD = 3; // fourth entry into "fields" is the "w velocity" field. Set the havg to zero for WFLD

    // Get a list of tracer names for retrieval
    std::vector<std::string> tracer_names = coupler.get_tracer_names();
    int num_tracers = coupler.get_num_tracers();

    auto &dm = coupler.get_data_manager_readwrite();

    // Create MultiField of all state and tracer full variables, since we're doing the same operation on each
    core::MultiField<real,3> full_fields;
    full_fields.add_field( dm.get<real,3>("density_dry") );
    full_fields.add_field( dm.get<real,3>("uvel"       ) );
    full_fields.add_field( dm.get<real,3>("vvel"       ) );
    full_fields.add_field( dm.get<real,3>("wvel"       ) );
    full_fields.add_field( dm.get<real,3>("temp"       ) );
    for (int tr=0; tr < num_tracers; tr++) { full_fields.add_field( dm.get<real,3>(tracer_names[tr]) ); }

    int num_fields = full_fields.get_num_fields();

    // Compute the horizontal average for each vertical level (that we use for the sponge layer)
    real2d havg_fields("havg_fields",num_fields,num_layers);
    havg_fields = 0;
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,num_layers,ny,nx) ,
                                      YAKL_LAMBDA (int ifld, int kloc, int j, int i) {
      int k = nz - 1 - kloc;
      if (ifld != WFLD) yakl::atomicAdd( havg_fields(ifld,kloc) , full_fields(ifld,k,j,i) );
    });

    havg_fields = coupler.get_parallel_comm().all_reduce( havg_fields , MPI_SUM , "sponge_Allreduce" );

    real time_factor = dt / time_scale;

    // use a cosine relaxation in space:  ((cos(pi*rel_dist)+1)/2)^2
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,num_layers,ny,nx) ,
                                      YAKL_LAMBDA (int ifld, int kloc, int j, int i) {
      int k = nz - 1 - kloc;
      real z = (k+0.5_fp)*dz;
      real rel_dist = ( zlen - z ) / ( num_layers * dz );
      real space_factor = ( cos(M_PI*rel_dist) + 1 ) / 2;
      real factor = space_factor * time_factor;
      full_fields(ifld,k,j,i) += ( havg_fields(ifld,kloc)/(nx_glob*ny_glob) - full_fields(ifld,k,j,i) ) * factor;
    });
  }

}
