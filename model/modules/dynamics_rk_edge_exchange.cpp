
#include "dynamics_rk.h"

namespace modules {

  void Dynamics_Euler_Stratified_WenoFV::edge_exchange( core::Coupler const & coupler           ,
                                                        real5d        const & state_limits_x    ,
                                                        real5d        const & tracers_limits_x  ,
                                                        real4d        const & pressure_limits_x ,
                                                        real5d        const & state_limits_y    ,
                                                        real5d        const & tracers_limits_y  ,
                                                        real4d        const & pressure_limits_y ,
                                                        real5d        const & state_limits_z    ,
                                                        real5d        const & tracers_limits_z  ,
                                                        real4d        const & pressure_limits_z ) const {
    #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_start("edge_exchange");
    #endif
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nx             = coupler.get_nx();
    auto ny             = coupler.get_ny();
    auto nz             = coupler.get_nz();
    auto num_tracers    = coupler.get_num_tracers();
    auto &neigh         = coupler.get_neighbor_rankid_matrix();
    auto dtype          = coupler.get_mpi_data_type();
    auto bc_z           = coupler.get_option<std::string>("bc_z","solid_wall");
    auto comm           = MPI_COMM_WORLD;
    auto &dm            = coupler.get_data_manager_readonly();
    auto hy_dens_edges  = dm.get<real const,1>("hy_dens_edges");
    auto hy_theta_edges = dm.get<real const,1>("hy_theta_edges");
    int npack = num_state + num_tracers+1;
    MPI_Request sReq [2];
    MPI_Request rReq [2];
    MPI_Status  sStat[2];
    MPI_Status  rStat[2];

    // x-exchange
    {
      real3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
      real3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);
      real3d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny);
      real3d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
        if        (v < num_state) {
          edge_send_buf_W(v,k,j) = state_limits_x  (1,v          ,k,j,0 );
          edge_send_buf_E(v,k,j) = state_limits_x  (0,v          ,k,j,nx);
        } else if (v < num_state + num_tracers) {                    
          edge_send_buf_W(v,k,j) = tracers_limits_x(1,v-num_state,k,j,0 );
          edge_send_buf_E(v,k,j) = tracers_limits_x(0,v-num_state,k,j,nx);
        } else {
          edge_send_buf_W(v,k,j) = pressure_limits_x(1,k,j,0 );
          edge_send_buf_E(v,k,j) = pressure_limits_x(0,k,j,nx);
        }
      });
      #ifdef PORTURB_GPU_AWARE_MPI
        yakl::timer_start("edge_exchange_mpi_x_gpu_aware");
        yakl::fence();
        MPI_Irecv( edge_recv_buf_W.data() , edge_recv_buf_W.size() , dtype , neigh(1,0) , 4 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E.data() , edge_recv_buf_E.size() , dtype , neigh(1,2) , 5 , comm , &rReq[1] );
        MPI_Isend( edge_send_buf_W.data() , edge_send_buf_W.size() , dtype , neigh(1,0) , 5 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_E.data() , edge_send_buf_E.size() , dtype , neigh(1,2) , 4 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        yakl::timer_stop("edge_exchange_mpi_x_gpu_aware");
      #else
        yakl::timer_start("edge_exchange_mpi_x");
        realHost3d edge_send_buf_W_host("edge_send_buf_W_host",npack,nz,ny);
        realHost3d edge_send_buf_E_host("edge_send_buf_E_host",npack,nz,ny);
        realHost3d edge_recv_buf_W_host("edge_recv_buf_W_host",npack,nz,ny);
        realHost3d edge_recv_buf_E_host("edge_recv_buf_E_host",npack,nz,ny);
        MPI_Irecv( edge_recv_buf_W_host.data() , edge_recv_buf_W_host.size() , dtype , neigh(1,0) , 4 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E_host.data() , edge_recv_buf_E_host.size() , dtype , neigh(1,2) , 5 , comm , &rReq[1] );
        edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
        edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
        yakl::fence();
        MPI_Isend( edge_send_buf_W_host.data() , edge_send_buf_W_host.size() , dtype , neigh(1,0) , 5 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_E_host.data() , edge_send_buf_E_host.size() , dtype , neigh(1,2) , 4 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        yakl::timer_stop("edge_exchange_mpi_x");
        edge_recv_buf_W_host.deep_copy_to(edge_recv_buf_W);
        edge_recv_buf_E_host.deep_copy_to(edge_recv_buf_E);
      #endif
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
        if        (v < num_state) {
          state_limits_x  (0,v          ,k,j,0 ) = edge_recv_buf_W(v,k,j);
          state_limits_x  (1,v          ,k,j,nx) = edge_recv_buf_E(v,k,j);
        } else if (v < num_state + num_tracers) {
          tracers_limits_x(0,v-num_state,k,j,0 ) = edge_recv_buf_W(v,k,j);
          tracers_limits_x(1,v-num_state,k,j,nx) = edge_recv_buf_E(v,k,j);
        } else {
          pressure_limits_x(0,k,j,0 ) = edge_recv_buf_W(v,k,j);
          pressure_limits_x(1,k,j,nx) = edge_recv_buf_E(v,k,j);
        }
      });
    }

    // y-direction exchange
    {
      real3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
      real3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);
      real3d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx);
      real3d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
        if        (v < num_state) {
          edge_send_buf_S(v,k,i) = state_limits_y  (1,v          ,k,0 ,i);
          edge_send_buf_N(v,k,i) = state_limits_y  (0,v          ,k,ny,i);
        } else if (v < num_state + num_tracers) {                    
          edge_send_buf_S(v,k,i) = tracers_limits_y(1,v-num_state,k,0 ,i);
          edge_send_buf_N(v,k,i) = tracers_limits_y(0,v-num_state,k,ny,i);
        } else {
          edge_send_buf_S(v,k,i) = pressure_limits_y(1,k,0 ,i);
          edge_send_buf_N(v,k,i) = pressure_limits_y(0,k,ny,i);
        }
      });
      #ifdef PORTURB_GPU_AWARE_MPI
        yakl::timer_start("edge_exchange_mpi_y_gpu_aware");
        yakl::fence();
        MPI_Irecv( edge_recv_buf_S.data() , edge_recv_buf_S.size() , dtype , neigh(0,1) , 6 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_N.data() , edge_recv_buf_N.size() , dtype , neigh(2,1) , 7 , comm , &rReq[1] );
        MPI_Isend( edge_send_buf_S.data() , edge_send_buf_S.size() , dtype , neigh(0,1) , 7 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_N.data() , edge_send_buf_N.size() , dtype , neigh(2,1) , 6 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        yakl::timer_stop("edge_exchange_mpi_y_gpu_aware");
      #else
        yakl::timer_start("edge_exchange_mpi_y");
        realHost3d edge_send_buf_S_host("edge_send_buf_S_host",npack,nz,nx);
        realHost3d edge_send_buf_N_host("edge_send_buf_N_host",npack,nz,nx);
        realHost3d edge_recv_buf_S_host("edge_recv_buf_S_host",npack,nz,nx);
        realHost3d edge_recv_buf_N_host("edge_recv_buf_N_host",npack,nz,nx);
        MPI_Irecv( edge_recv_buf_S_host.data() , edge_recv_buf_S_host.size() , dtype , neigh(0,1) , 6 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_N_host.data() , edge_recv_buf_N_host.size() , dtype , neigh(2,1) , 7 , comm , &rReq[1] );
        edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
        edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
        yakl::fence();
        MPI_Isend( edge_send_buf_S_host.data() , edge_send_buf_S_host.size() , dtype , neigh(0,1) , 7 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_N_host.data() , edge_send_buf_N_host.size() , dtype , neigh(2,1) , 6 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        yakl::timer_stop("edge_exchange_mpi_y");
        edge_recv_buf_S_host.deep_copy_to(edge_recv_buf_S);
        edge_recv_buf_N_host.deep_copy_to(edge_recv_buf_N);
      #endif
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
        if        (v < num_state) {
          state_limits_y  (0,v          ,k,0 ,i) = edge_recv_buf_S(v,k,i);
          state_limits_y  (1,v          ,k,ny,i) = edge_recv_buf_N(v,k,i);
        } else if (v < num_state + num_tracers) {
          tracers_limits_y(0,v-num_state,k,0 ,i) = edge_recv_buf_S(v,k,i);
          tracers_limits_y(1,v-num_state,k,ny,i) = edge_recv_buf_N(v,k,i);
        } else {
          pressure_limits_y(0,k,0 ,i) = edge_recv_buf_S(v,k,i);
          pressure_limits_y(1,k,ny,i) = edge_recv_buf_N(v,k,i);
        }
      });
    }

    if (bc_z == "solid_wall") {
      // z-direction BC's
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
        // Dirichlet
        state_limits_z(0,idR,0 ,j,i) = hy_dens_edges(0);
        state_limits_z(1,idR,0 ,j,i) = hy_dens_edges(0);
        state_limits_z(0,idW,0 ,j,i) = 0;
        state_limits_z(1,idW,0 ,j,i) = 0;
        state_limits_z(0,idT,0 ,j,i) = hy_theta_edges(0);
        state_limits_z(1,idT,0 ,j,i) = hy_theta_edges(0);
        state_limits_z(0,idR,nz,j,i) = hy_dens_edges(nz);
        state_limits_z(1,idR,nz,j,i) = hy_dens_edges(nz);
        state_limits_z(0,idW,nz,j,i) = 0;
        state_limits_z(1,idW,nz,j,i) = 0;
        state_limits_z(0,idT,nz,j,i) = hy_theta_edges(nz);
        state_limits_z(1,idT,nz,j,i) = hy_theta_edges(nz);
        for (int l=0; l < num_tracers; l++) {
          tracers_limits_z(0,l,0 ,j,i) = 0;
          tracers_limits_z(1,l,0 ,j,i) = 0;
          tracers_limits_z(0,l,nz,j,i) = 0;
          tracers_limits_z(1,l,nz,j,i) = 0;
        }
        // Neumann
        state_limits_z   (0,idU,0 ,j,i) = state_limits_z   (1,idU,0 ,j,i);
        state_limits_z   (0,idV,0 ,j,i) = state_limits_z   (1,idV,0 ,j,i);
        pressure_limits_z(0    ,0 ,j,i) = pressure_limits_z(1    ,0 ,j,i);
        state_limits_z   (1,idU,nz,j,i) = state_limits_z   (0,idU,nz,j,i);
        state_limits_z   (1,idV,nz,j,i) = state_limits_z   (0,idV,nz,j,i);
        pressure_limits_z(1    ,nz,j,i) = pressure_limits_z(0    ,nz,j,i);
      });
    } else if (bc_z == "periodic") {
      // z-direction BC's
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
        state_limits_z   (0,idR,0 ,j,i) = state_limits_z   (0,idR,nz,j,i);
        state_limits_z   (0,idU,0 ,j,i) = state_limits_z   (0,idU,nz,j,i);
        state_limits_z   (0,idV,0 ,j,i) = state_limits_z   (0,idV,nz,j,i);
        state_limits_z   (0,idW,0 ,j,i) = state_limits_z   (0,idW,nz,j,i);
        state_limits_z   (0,idT,0 ,j,i) = state_limits_z   (0,idT,nz,j,i);
        pressure_limits_z(0    ,0 ,j,i) = pressure_limits_z(0    ,nz,j,i);
        state_limits_z   (1,idR,nz,j,i) = state_limits_z   (1,idR,0 ,j,i);
        state_limits_z   (1,idU,nz,j,i) = state_limits_z   (1,idU,0 ,j,i);
        state_limits_z   (1,idV,nz,j,i) = state_limits_z   (1,idV,0 ,j,i);
        state_limits_z   (1,idW,nz,j,i) = state_limits_z   (1,idW,0 ,j,i);
        state_limits_z   (1,idT,nz,j,i) = state_limits_z   (1,idT,0 ,j,i);
        pressure_limits_z(1    ,nz,j,i) = pressure_limits_z(1    ,0 ,j,i);
        for (int l=0; l < num_tracers; l++) {
          tracers_limits_z(0,l,0 ,j,i) = tracers_limits_z(0,l,nz,j,i);
          tracers_limits_z(1,l,nz,j,i) = tracers_limits_z(1,l,0 ,j,i);
        }
      });
    }
    #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("edge_exchange");
    #endif
  }

}


