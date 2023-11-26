
#pragma once

namespace custom_modules {

  struct VelocityHistogram {
    int static constexpr num_bins =  80;
    int static constexpr lb       = -40;
    int static constexpr ub       =  40;
    yakl::Array<unsigned int,5,yakl::memDevice,yakl::styleC> uvel, vvel, wvel;

    void init( core::Coupler const &coupler ) {
      auto nx   = coupler.get_nx  ();
      auto ny   = coupler.get_ny  ();
      auto nz   = coupler.get_nz  ();
      auto nens = coupler.get_nens();
      uvel = yakl::Array<unsigned int,5,yakl::memDevice,yakl::styleC>("uvel_hist",num_bins,nz,ny,nx,nens);
      vvel = yakl::Array<unsigned int,5,yakl::memDevice,yakl::styleC>("vvel_hist",num_bins,nz,ny,nx,nens);
      wvel = yakl::Array<unsigned int,5,yakl::memDevice,yakl::styleC>("wvel_hist",num_bins,nz,ny,nx,nens);
      uvel = 0;
      vvel = 0;
      wvel = 0;
    }

    void update( core::Coupler const &coupler ) {
      auto nx   = coupler.get_nx  ();
      auto ny   = coupler.get_ny  ();
      auto nz   = coupler.get_nz  ();
      auto nens = coupler.get_nens();
      auto dm_uvel = coupler.get_data_manager_readonly().get<real const,4>("uvel");
      auto dm_vvel = coupler.get_data_manager_readonly().get<real const,4>("vvel");
      auto dm_wvel = coupler.get_data_manager_readonly().get<real const,4>("wvel");
      YAKL_SCOPE( uvel , this->uvel );
      YAKL_SCOPE( vvel , this->vvel );
      YAKL_SCOPE( wvel , this->wvel );
      yakl::c::parallel_for( YAKL_AUTO_LABEL() ,
                             yakl::c::SimpleBounds<4>(nz,ny,nx,nens) ,
                             YAKL_LAMBDA (int k, int j, int i, int iens) {
        uvel(std::min(num_bins-1,std::max(0,static_cast<int>(std::floor(dm_uvel(k,j,i,iens)+100)))),k,j,i,iens)++;
        vvel(std::min(num_bins-1,std::max(0,static_cast<int>(std::floor(dm_vvel(k,j,i,iens)+100)))),k,j,i,iens)++;
        wvel(std::min(num_bins-1,std::max(0,static_cast<int>(std::floor(dm_wvel(k,j,i,iens)+100)))),k,j,i,iens)++;
      });
    }

    void dump( core::Coupler const &coupler ) {
      using yakl::c::parallel_for;
      yakl::timer_start("histogram_output");
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dx          = coupler.get_dx();
      auto dy          = coupler.get_dy();
      auto dz          = coupler.get_dz();
      int i_beg        = coupler.get_i_beg();
      int j_beg        = coupler.get_j_beg();
      yakl::SimpleNetCDF nc;
      int rank = 0;
      MPI_Comm_rank( MPI_COMM_WORLD , &rank );
      std::stringstream fname;
      fname << "histogram_" << std::setw(6) << std::setfill('0') << rank << ".nc";
      nc.create( fname.str() , yakl::NETCDF_MODE_REPLACE );
      real1d bin ("bin" ,num_bins);
      real1d ens ("ens" ,nens    );
      real1d xloc("xloc",nx      );
      real1d yloc("yloc",ny      );
      real1d zloc("zloc",nz      );
      parallel_for( YAKL_AUTO_LABEL() , nens     , YAKL_LAMBDA (int iens) { ens (iens) = iens;                });
      parallel_for( YAKL_AUTO_LABEL() , nx       , YAKL_LAMBDA (int i   ) { xloc(i   ) = (i_beg+i+0.5_fp)*dx; });
      parallel_for( YAKL_AUTO_LABEL() , ny       , YAKL_LAMBDA (int j   ) { yloc(j   ) = (j_beg+j+0.5_fp)*dy; });
      parallel_for( YAKL_AUTO_LABEL() , nz       , YAKL_LAMBDA (int k   ) { zloc(k   ) = (      k+0.5_fp)*dz; });
      parallel_for( YAKL_AUTO_LABEL() , num_bins , YAKL_LAMBDA (int ibin) { bin (ibin) = ibin;                });
      nc.write(ens .createHostCopy(),"ens",{"ens"});
      nc.write(xloc.createHostCopy(),"x"  ,{"x"  });
      nc.write(yloc.createHostCopy(),"y"  ,{"y"  });
      nc.write(zloc.createHostCopy(),"z"  ,{"z"  });
      nc.write(bin .createHostCopy(),"bin",{"bin"});
      nc.write(uvel.createHostCopy(),"uvel_hist",{"bin","z","y","x","ens"});
      nc.write(vvel.createHostCopy(),"vvel_hist",{"bin","z","y","x","ens"});
      nc.write(wvel.createHostCopy(),"wvel_hist",{"bin","z","y","x","ens"});
      nc.close();
      yakl::timer_stop("histogram_output");
    }

  };

}


