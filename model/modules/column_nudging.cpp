
#include "column_nudging.h"

namespace modules {

void ColumnNudger::set_column( core::Coupler &coupler , std::vector<std::string> names_in ) {
  using yakl::c::parallel_for;
  using yakl::c::SimpleBounds;
  int nx   = coupler.get_nx();
  int ny   = coupler.get_ny();
  int nz   = coupler.get_nz();
  names = names_in;
  column = real2d("column",names.size(),nz);
  auto &dm = coupler.get_data_manager_readonly();
  core::MultiField<real const,3> state;
  for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real const,3>(names[i]) ); }
  column = get_column_average( coupler , state );
}


void ColumnNudger::nudge_to_column( core::Coupler &coupler , real dt , real time_scale) {
  using yakl::c::parallel_for;
  using yakl::c::SimpleBounds;
  int nx   = coupler.get_nx();
  int ny   = coupler.get_ny();
  int nz   = coupler.get_nz();
  auto &dm = coupler.get_data_manager_readwrite();
  auto immersed = dm.get<real const,3>("immersed_proportion");
  core::MultiField<real,3> state;
  for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real,3>(names[i]) ); }
  auto state_col_avg = get_column_average( coupler , state );
  YAKL_SCOPE( column , this->column );
  parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                    YAKL_LAMBDA (int l, int k, int j, int i) {
    if (immersed(k,j,i) == 0) {
      state(l,k,j,i) += dt * ( column(l,k) - state_col_avg(l,k) ) / time_scale;
    }
  });
}


template <class T>
real2d ColumnNudger::get_column_average( core::Coupler const &coupler , core::MultiField<T,3> &state ) const {
  using yakl::c::parallel_for;
  using yakl::c::SimpleBounds;
  int nx_glob = coupler.get_nx_glob();
  int ny_glob = coupler.get_ny_glob();
  int nx      = coupler.get_nx();
  int ny      = coupler.get_ny();
  int nz      = coupler.get_nz();
  real2d column_loc("column_loc",names.size(),nz);
  column_loc = 0;
  parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                    YAKL_LAMBDA (int l, int k, int j, int i) {
    yakl::atomicAdd( column_loc(l,k) , state(l,k,j,i) );
  });
  #ifdef MW_GPU_AWARE_MPI
    auto column_total = column_loc.createDeviceObject();
    yakl::fence();
    yakl::timer_start("column_nudging_Allreduce");
    MPI_Allreduce( column_loc.data() , column_total.data() , column_total.size() ,
                   coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
    yakl::timer_stop("column_nudging_Allreduce");
  #else
    yakl::timer_start("column_nudging_Allreduce");
    auto column_total_host = column_loc.createHostObject();
    MPI_Allreduce( column_loc.createHostCopy().data() , column_total_host.data() , column_total_host.size() ,
                   coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
    auto column_total = column_total_host.createDeviceCopy();
    yakl::timer_stop("column_nudging_Allreduce");
  #endif
  parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) , YAKL_LAMBDA (int l, int k) {
    column_loc(l,k) = column_total(l,k) / (nx_glob*ny_glob);
  });
  return column_loc;
}

}


