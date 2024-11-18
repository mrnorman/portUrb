!wrf:model_layer:physics
!

! this module contains the two-moment microphysics code described by
!     morrison et al. (2009, mwr)

! changes for v3.2, relative to most recent (bug-fix) code for v3.1

! 1) added accelerated melting of graupel/snow due to collision with rain, following lin et al. (1983)
! 2) increased minimum lambda for rain, and added rain drop breakup following modified version
!     of verlinde and cotton (1993)
! 3) change minimum allowed mixing ratios in dry conditions (rh < 90%), this improves radar reflectiivity
!     in low reflectivity regions
! 4) bug fix to maximum allowed particle fallspeeds as a function of air density
! 5) bug fix to calculation of liquid water saturation vapor pressure (change is very minor)
! 6) include wrf constants per suggestion of jimy

! bug fix, 5/12/10
! 7) bug fix for saturation vapor pressure in low pressure, to avoid division by zero
! 8) include 'ep2' wrf constant for saturation mixing ratio calculation, instead of hardwire constant

! changes for v3.3
! 1) modification for coupling with wrf-chem (predicted droplet number concentration) as an option
! 2) modify fallspeed below the lowest level of precipitation, which prevents
!      potential for spurious accumulation of precipitation during sub-stepping for sedimentation
! 3) bug fix to latent heat release due to collisions of cloud ice with rain
! 4) clean up of comments in the code
    
! additional minor bug fixes and small changes, 5/30/2011
! minor revisions by a. ackerman april 2011:
! 1) replaced kinematic with dynamic viscosity 
! 2) replaced scaling by air density for cloud droplet sedimentation
!    with viscosity-dependent stokes expression
! 3) use ikawa and saito (1991) air-density scaling for cloud ice
! 4) corrected typo in 2nd digit of ventilation constant f2r

! additional fixes:
! 5) temperature for accelerated melting due to colliions of snow and graupel
!    with rain should use celsius, not kelvin (bug reported by k. van weverberg)
! 6) npracs is not subtracted from snow number concentration, since
!    decrease in snow number is already accounted for by nsmlts 
! 7) fix for switch for running w/o graupel/hail (cloud ice and snow only)

! hm bug fix 3/16/12

! 1) very minor change to limits on autoconversion source of rain number when cloud water is depleted

! wrfv3.5
! hm/a. ackerman bug fix 11/08/12

! 1) for accelerated melting from collisions, should use rain mass collected by snow, not snow mass 
!    collected by rain
! 2) minor changes to some comments
! 3) reduction of maximum-allowed ice concentration from 10 cm-3 to 0.3
!    cm-3. this was done to address the problem of excessive and persistent
!    anvil cirrus produced by the scheme.

! changes for wrfv3.5.1
! 1) added output for snow+cloud ice and graupel time step and accumulated
!    surface precipitation
! 2) bug fix to option w/o graupel/hail (igraup = 1), include praci, pgsacw,
!    and pgracs as sources for snow instead of graupel/hail, bug reported by
!    hailong wang (pnnl)
! 3) very minor fix to immersion freezing rate formulation (negligible impact)
! 4) clarifications to code comments
! 5) minor change to shedding of rain, remove limit so that the number of 
!    collected drops can smaller than number of shed drops
! 6) change of specific heat of liquid water from 4218 to 4187 j/kg/k

! changes for wrfv3.6.1
! 1) minor bug fix to melting of snow and graupel, an extra factor of air density (rho) was removed
!    from the calculation of psmlt and pgmlt
! 2) redundant initialization of psmlt (non answer-changing)

! changes for wrfv3.8.1
! 1) changes and cleanup of code comments
! 2) correction to universal gas constant (very small change)

! changes for wrfv4.3
! 1) fix to saturation vapor pressure polysvp to work at t < -80 c
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! this scheme is a bulk double-moment scheme that predicts mixing
! ratios and number concentrations of five hydrometeor species:
! cloud droplets, cloud (small) ice, rain, snow, and graupel/hail.

module module_mp_morr_two_moment
   use     module_wrf_error
   use module_mp_radar

!  use wrf physics constants
  use module_model_constants, only: cp, g, r => r_d, rv => r_v, ep_2

!  use module_state_description

   implicit none

   real, parameter :: pi = 3.1415926535897932384626434
   real, parameter :: xxx = 0.9189385332046727417803297

   public  ::  mp_morr_two_moment
   public  ::  polysvp

   private :: gamma, derf1
   private :: pi, xxx
   private :: morr_two_moment_micro

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! switches for microphysics scheme
! iact = 1, use power-law ccn spectra, nccn = cs^k
! iact = 2, use lognormal aerosol size dist to derive ccn spectra
! iact = 3, activation calculated in module_mixactivate

     integer, private ::  iact

! inum = 0, predict droplet concentration
! inum = 1, assume constant droplet concentration   
! !!!note: predicted droplet concentration not available in this version
! contact hugh morrison (morrison@ucar.edu) for further information

     integer, private ::  inum

! for inum = 1, set constant droplet concentration (cm-3)
     real, private ::      ndcnst

! switch for liquid-only run
! iliq = 0, include ice
! iliq = 1, liquid only, no ice

     integer, private ::  iliq

! switch for ice nucleation
! inuc = 0, use formula from rasmussen et al. 2002 (mid-latitude)
!      = 1, use mpace observations

     integer, private ::  inuc

! ibase = 1, neglect droplet activation at lateral cloud edges due to 
!             unresolved entrainment and mixing, activate
!             at cloud base or in region with little cloud water using 
!             non-equlibrium supersaturation, 
!             in cloud interior activate using equilibrium supersaturation
! ibase = 2, assume droplet activation at lateral cloud edges due to 
!             unresolved entrainment and mixing dominates,
!             activate droplets everywhere in the cloud using non-equilibrium
!             supersaturation, based on the 
!             local sub-grid and/or grid-scale vertical velocity 
!             at the grid point

! note: only used for predicted droplet concentration (inum = 0) in non-wrf-chem version of code

     integer, private ::  ibase

! include sub-grid vertical velocity in droplet activation
! isub = 0, include sub-grid w (recommended for lower resolution)
! isub = 1, exclude sub-grid w, only use grid-scale w

! note: only used for predicted droplet concentration (inum = 0) in non-wrf-chem version of code

     integer, private ::  isub      

! switch for graupel/no graupel
! igraup = 0, include graupel
! igraup = 1, no graupel

     integer, private ::  igraup

! hm added new option for hail
! switch for hail/graupel
! ihail = 0, dense precipitating ice is graupel
! ihail = 1, dense precipitating gice is hail

     integer, private ::  ihail

! cloud microphysics constants

     real, private ::      ai,ac,as,ar,ag ! 'a' parameter in fallspeed-diam relationship
     real, private ::      bi,bc,bs,br,bg ! 'b' parameter in fallspeed-diam relationship
!     real, private ::      r           ! gas constant for air
!     real, private ::      rv          ! gas constant for water vapor
!     real, private ::      cp          ! specific heat at constant pressure for dry air
     real, private ::      rhosu       ! standard air density at 850 mb
     real, private ::      rhow        ! density of liquid water
     real, private ::      rhoi        ! bulk density of cloud ice
     real, private ::      rhosn       ! bulk density of snow
     real, private ::      rhog        ! bulk density of graupel
     real, private ::      aimm        ! parameter in bigg immersion freezing
     real, private ::      bimm        ! parameter in bigg immersion freezing
     real, private ::      ecr         ! collection efficiency between droplets/rain and snow/rain
     real, private ::      dcs         ! threshold size for cloud ice autoconversion
     real, private ::      mi0         ! initial size of nucleated crystal
     real, private ::      mg0         ! mass of embryo graupel
     real, private ::      f1s         ! ventilation parameter for snow
     real, private ::      f2s         ! ventilation parameter for snow
     real, private ::      f1r         ! ventilation parameter for rain
     real, private ::      f2r         ! ventilation parameter for rain
!     real, private ::      g           ! gravitational acceleration
     real, private ::      qsmall      ! smallest allowed hydrometeor mixing ratio
     real, private ::      ci,di,cs,ds,cg,dg ! size distribution parameters for cloud ice, snow, graupel
     real, private ::      eii         ! collection efficiency, ice-ice collisions
     real, private ::      eci         ! collection efficiency, ice-droplet collisions
     real, private ::      rin     ! radius of contact nuclei (m)
! hm, add for v3.2
     real, private ::      cpw     ! specific heat of liquid water

! ccn spectra for iact = 1

     real, private ::      c1     ! 'c' in nccn = cs^k (cm-3)
     real, private ::      k1     ! 'k' in nccn = cs^k

! aerosol parameters for iact = 2

     real, private ::      mw      ! molecular weight water (kg/mol)
     real, private ::      osm     ! osmotic coefficient
     real, private ::      vi      ! number of ion dissociated in solution
     real, private ::      epsm    ! aerosol soluble fraction
     real, private ::      rhoa    ! aerosol bulk density (kg/m3)
     real, private ::      map     ! molecular weight aerosol (kg/mol)
     real, private ::      ma      ! molecular weight of 'air' (kg/mol)
     real, private ::      rr      ! universal gas constant
     real, private ::      bact    ! activation parameter
     real, private ::      rm1     ! geometric mean radius, mode 1 (m)
     real, private ::      rm2     ! geometric mean radius, mode 2 (m)
     real, private ::      nanew1  ! total aerosol concentration, mode 1 (m^-3)
     real, private ::      nanew2  ! total aerosol concentration, mode 2 (m^-3)
     real, private ::      sig1    ! standard deviation of aerosol s.d., mode 1
     real, private ::      sig2    ! standard deviation of aerosol s.d., mode 2
     real, private ::      f11     ! correction factor for activation, mode 1
     real, private ::      f12     ! correction factor for activation, mode 1
     real, private ::      f21     ! correction factor for activation, mode 2
     real, private ::      f22     ! correction factor for activation, mode 2     
     real, private ::      mmult   ! mass of splintered ice particle
     real, private ::      lammaxi,lammini,lammaxr,lamminr,lammaxs,lammins,lammaxg,lamming

! constants to improve efficiency

     real, private :: cons1,cons2,cons3,cons4,cons5,cons6,cons7,cons8,cons9,cons10
     real, private :: cons11,cons12,cons13,cons14,cons15,cons16,cons17,cons18,cons19,cons20
     real, private :: cons21,cons22,cons23,cons24,cons25,cons26,cons27,cons28,cons29,cons30
     real, private :: cons31,cons32,cons33,cons34,cons35,cons36,cons37,cons38,cons39,cons40
     real, private :: cons41


contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine morr_two_moment_init(morr_rimed_ice) bind(c,name="morr_two_moment_init") ! ras  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! this subroutine initializes all physical constants amnd parameters 
! needed by the microphysics scheme.
! needs to be called at first time step, prior to call to main microphysics interface
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      integer, intent(in):: morr_rimed_ice ! ras  

      integer n,i

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! the following parameters are user-defined switches and need to be
! set prior to code compilation

! inum is automatically set to 0 for wrf-chem below,
! allowing prediction of droplet concentration
! thus, this parameter should not be changed here
! and should be left to 1

      inum = 1

! set constant droplet concentration (units of cm-3)
! if no coupling with wrf-chem

      ndcnst = 250.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! note, the following options related to droplet activation 
! (iact, ibase, isub) are not available in current version
! for wrf-chem, droplet activation is performed 
! in 'mix_activate', not in microphysics scheme


! iact = 1, use power-law ccn spectra, nccn = cs^k
! iact = 2, use lognormal aerosol size dist to derive ccn spectra

      iact = 2

! ibase = 1, neglect droplet activation at lateral cloud edges due to 
!             unresolved entrainment and mixing, activate
!             at cloud base or in region with little cloud water using 
!             non-equlibrium supersaturation assuming no initial cloud water, 
!             in cloud interior activate using equilibrium supersaturation
! ibase = 2, assume droplet activation at lateral cloud edges due to 
!             unresolved entrainment and mixing dominates,
!             activate droplets everywhere in the cloud using non-equilibrium
!             supersaturation assuming no initial cloud water, based on the 
!             local sub-grid and/or grid-scale vertical velocity 
!             at the grid point

! note: only used for predicted droplet concentration (inum = 0)
 
      ibase = 2

! include sub-grid vertical velocity (standard deviation of w) in droplet activation
! isub = 0, include sub-grid w (recommended for lower resolution)
! currently, sub-grid w is constant of 0.5 m/s (not coupled with pbl/turbulence scheme)
! isub = 1, exclude sub-grid w, only use grid-scale w

! note: only used for predicted droplet concentration (inum = 0)

      isub = 0      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! switch for liquid-only run
! iliq = 0, include ice
! iliq = 1, liquid only, no ice

      iliq = 0

! switch for ice nucleation
! inuc = 0, use formula from rasmussen et al. 2002 (mid-latitude)
!      = 1, use mpace observations (arctic only)

      inuc = 0

! switch for graupel/hail no graupel/hail
! igraup = 0, include graupel/hail
! igraup = 1, no graupel/hail

      igraup = 0

! hm added 11/7/07
! switch for hail/graupel
! ihail = 0, dense precipitating ice is graupel
! ihail = 1, dense precipitating ice is hail
! note ---> recommend ihail = 1 for continental deep convection

      !ihail = 0 !changed to namelist option (morr_rimed_ice) by ras
      ! check if namelist option is feasible, otherwise default to graupel - ras
      if (morr_rimed_ice .eq. 1) then
         ihail = 1
      else
         ihail = 0
      endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! set physical constants

! fallspeed parameters (v=ad^b)
         ai = 700.
         ac = 3.e7
         as = 11.72
         ar = 841.99667
         bi = 1.
         bc = 2.
         bs = 0.41
         br = 0.8
         if (ihail.eq.0) then
	 ag = 19.3
	 bg = 0.37
         else ! (matsun and huggins 1980)
         ag = 114.5 
         bg = 0.5
         end if

! constants and parameters
!         r = 287.15
!         rv = 461.5
!         cp = 1005.
         rhosu = 85000./(287.15*273.15)
         rhow = 997.
         rhoi = 500.
         rhosn = 100.
         if (ihail.eq.0) then
	 rhog = 400.
         else
         rhog = 900.
         end if
         aimm = 0.66
         bimm = 100.
         ecr = 1.
         dcs = 125.e-6
         mi0 = 4./3.*pi*rhoi*(10.e-6)**3
	 mg0 = 1.6e-10
         f1s = 0.86
         f2s = 0.28
         f1r = 0.78
!         f2r = 0.32
! fix 053011
         f2r = 0.308
!         g = 9.806
         qsmall = 1.e-14
         eii = 0.1
         eci = 0.7
! hm, add for v3.2
! hm, 7/23/13
!         cpw = 4218.
         cpw = 4187.

! size distribution parameters

         ci = rhoi*pi/6.
         di = 3.
         cs = rhosn*pi/6.
         ds = 3.
         cg = rhog*pi/6.
         dg = 3.

! radius of contact nuclei
         rin = 0.1e-6

         mmult = 4./3.*pi*rhoi*(5.e-6)**3

! size limits for lambda

         lammaxi = 1./1.e-6
         lammini = 1./(2.*dcs+100.e-6)
         lammaxr = 1./20.e-6
!         lamminr = 1./500.e-6
         lamminr = 1./2800.e-6

         lammaxs = 1./10.e-6
         lammins = 1./2000.e-6
         lammaxg = 1./20.e-6
         lamming = 1./2000.e-6

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! note: these parameters only used by the non-wrf-chem version of the 
!       scheme with predicted droplet number

! ccn spectra for iact = 1

! maritime
! modified from rasmussen et al. 2002
! nccn = c*s^k, nccn is in cm-3, s is supersaturation ratio in %

              k1 = 0.4
              c1 = 120. 

! continental

!              k1 = 0.5
!              c1 = 1000. 

! aerosol activation parameters for iact = 2
! parameters currently set for ammonium sulfate

         mw = 0.018
         osm = 1.
         vi = 3.
         epsm = 0.7
         rhoa = 1777.
         map = 0.132
         ma = 0.0284
! hm fix 6/23/16
!         rr = 8.3187
         rr = 8.3145
         bact = vi*osm*epsm*mw*rhoa/(map*rhow)

! aerosol size distribution parameters currently set for mpace 
! (see morrison et al. 2007, jgr)
! mode 1

         rm1 = 0.052e-6
         sig1 = 2.04
         nanew1 = 72.2e6
         f11 = 0.5*exp(2.5*(log(sig1))**2)
         f21 = 1.+0.25*log(sig1)

! mode 2

         rm2 = 1.3e-6
         sig2 = 2.5
         nanew2 = 1.8e6
         f12 = 0.5*exp(2.5*(log(sig2))**2)
         f22 = 1.+0.25*log(sig2)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! constants for efficiency

         cons1=gamma(1.+ds)*cs
         cons2=gamma(1.+dg)*cg
         cons3=gamma(4.+bs)/6.
         cons4=gamma(4.+br)/6.
         cons5=gamma(1.+bs)
         cons6=gamma(1.+br)
         cons7=gamma(4.+bg)/6.
         cons8=gamma(1.+bg)
         cons9=gamma(5./2.+br/2.)
         cons10=gamma(5./2.+bs/2.)
         cons11=gamma(5./2.+bg/2.)
         cons12=gamma(1.+di)*ci
         cons13=gamma(bs+3.)*pi/4.*eci
         cons14=gamma(bg+3.)*pi/4.*eci
         cons15=-1108.*eii*pi**((1.-bs)/3.)*rhosn**((-2.-bs)/3.)/(4.*720.)
         cons16=gamma(bi+3.)*pi/4.*eci
         cons17=4.*2.*3.*rhosu*pi*eci*eci*gamma(2.*bs+2.)/(8.*(rhog-rhosn))
         cons18=rhosn*rhosn
         cons19=rhow*rhow
         cons20=20.*pi*pi*rhow*bimm
         cons21=4./(dcs*rhoi)
         cons22=pi*rhoi*dcs**3/6.
         cons23=pi/4.*eii*gamma(bs+3.)
         cons24=pi/4.*ecr*gamma(br+3.)
         cons25=pi*pi/24.*rhow*ecr*gamma(br+6.)
         cons26=pi/6.*rhow
         cons27=gamma(1.+bi)
         cons28=gamma(4.+bi)/6.
         cons29=4./3.*pi*rhow*(25.e-6)**3
         cons30=4./3.*pi*rhow
         cons31=pi*pi*ecr*rhosn
         cons32=pi/2.*ecr
         cons33=pi*pi*ecr*rhog
         cons34=5./2.+br/2.
         cons35=5./2.+bs/2.
         cons36=5./2.+bg/2.
         cons37=4.*pi*1.38e-23/(6.*pi*rin)
         cons38=pi*pi/3.*rhow
         cons39=pi*pi/36.*rhow*bimm
         cons40=pi/6.*bimm
         cons41=pi*pi*ecr*rhow

!+---+-----------------------------------------------------------------+
!..set these variables needed for computing radar reflectivity.  these
!.. get used within radar_init to create other variables used in the
!.. radar module.

         xam_r = pi*rhow/6.
         xbm_r = 3.
         xmu_r = 0.
         xam_s = cs
         xbm_s = ds
         xmu_s = 0.
         xam_g = cg
         xbm_g = dg
         xmu_g = 0.

         call radar_init
!+---+-----------------------------------------------------------------+


end subroutine morr_two_moment_init

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! this subroutine is main interface with the two-moment microphysics scheme
! this interface takes in 3d variables from driver model, converts to 1d for
! call to the main microphysics subroutine (subroutine morr_two_moment_micro) 
! which operates on 1d vertical columns.
! 1d variables from the main microphysics subroutine are then reassigned back to 3d for output
! back to driver model using this interface.
! microphysics tendencies are added to variables here before being passed back to driver model.

! this code was written by hugh morrison (ncar) and slava tatarskii (georgia tech).

! for questions, contact: hugh morrison, e-mail: morrison@ucar.edu, phone:303-497-8916

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine mp_morr_two_moment(itimestep,                       &
                th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, &
                rho, pii, p, dt_in, dz, w,          &
                rainnc, rainncv, sr,                    &
		snownc,snowncv,graupelnc,graupelncv,    & ! hm added 7/13/13
                refl_10cm, diagflag, do_radar_ref,      & ! gt added for reflectivity calcs
                qrcuten, qscuten, qicuten               & ! hm added
               ,f_qndrop, qndrop                        & ! hm added, wrf-chem 
               ,ids,ide, jds,jde, kds,kde               & ! domain dims
               ,ims,ime, jms,jme, kms,kme               & ! memory dims
               ,its,ite, jts,jte, kts,kte               & ! tile   dims            )
!jdf		   ,c2prec3d,csed3d,ised3d,ssed3d,gsed3d,rsed3d & ! hm add, wrf-chem
               ,wetscav_on, rainprod, evapprod                      &
		   ,qlsink,precr,preci,precs,precg &        ! hm add, wrf-chem
                                            ) bind(c,name="mp_morr_two_moment")
 
! qv - water vapor mixing ratio (kg/kg)
! qc - cloud water mixing ratio (kg/kg)
! qr - rain water mixing ratio (kg/kg)
! qi - cloud ice mixing ratio (kg/kg)
! qs - snow mixing ratio (kg/kg)
! qg - graupel mixing ratio (kg/kg)
! ni - cloud ice number concentration (1/kg)
! ns - snow number concentration (1/kg)
! nr - rain number concentration (1/kg)
! ng - graupel number concentration (1/kg)
! note: rho and ht not used by this scheme and do not need to be passed into scheme!!!!
! p - air pressure (pa)
! w - vertical air velocity (m/s)
! th - potential temperature (k)
! pii - exner function - used to convert potential temp to temp
! dz - difference in height over interface (m)
! dt_in - model time step (sec)
! itimestep - time step counter
! rainnc - accumulated grid-scale precipitation (mm)
! rainncv - one time step grid scale precipitation (mm/time step)
! snownc - accumulated grid-scale snow plus cloud ice (mm)
! snowncv - one time step grid scale snow plus cloud ice (mm/time step)
! graupelnc - accumulated grid-scale graupel (mm)
! graupelncv - one time step grid scale graupel (mm/time step)
! sr - one time step mass ratio of snow to total precip
! qrcuten, rain tendency from parameterized cumulus convection
! qscuten, snow tendency from parameterized cumulus convection
! qicuten, cloud ice tendency from parameterized cumulus convection

! variables below currently not in use, not coupled to pbl or radiation codes
! tke - turbulence kinetic energy (m^2 s-2), needed for droplet activation (see code below)
! nctend - droplet concentration tendency from pbl (kg-1 s-1)
! nctend - cloud ice concentration tendency from pbl (kg-1 s-1)
! kzh - heat eddy diffusion coefficient from ysu scheme (m^2 s-1), needed for droplet activation (see code below)
! effcs - cloud droplet effective radius output to radiation code (micron)
! effis - cloud droplet effective radius output to radiation code (micron)
! hm, added for wrf-chem coupling
! qlsink - tendency of cloud water to rain, snow, graupel (kg/kg/s)
! csed,ised,ssed,gsed,rsed - sedimentation fluxes (kg/m^2/s) for cloud water, ice, snow, graupel, rain
! preci,precs,precg,precr - sedimentation fluxes (kg/m^2/s) for ice, snow, graupel, rain

! rainprod - total tendency of conversion of cloud water/ice and graupel to rain (kg kg-1 s-1)
! evapprod - tendency of evaporation of rain (kg kg-1 s-1)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! reflectivity currently not included!!!!
! refl_10cm - calculated radar reflectivity at 10 cm (dbz)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! effc - droplet effective radius (micron)
! effr - rain effective radius (micron)
! effs - snow effective radius (micron)
! effi - cloud ice effective radius (micron)

! additional output from micro - sedimentation tendencies, needed for liquid-ice static energy

! qgsten - graupel sedimentation tend (kg/kg/s)
! qrsten - rain sedimentation tend (kg/kg/s)
! qisten - cloud ice sedimentation tend (kg/kg/s)
! qnisten - snow sedimentation tend (kg/kg/s)
! qcsten - cloud water sedimentation tend (kg/kg/s)

! wvar - standard deviation of sub-grid vertical velocity (m/s)

   use iso_c_binding, only: c_int, c_float
   implicit none

   integer(c_int),      intent(in   )    ::   ids, ide, jds, jde, kds, kde , &
                                       ims, ime, jms, jme, kms, kme , &
                                       its, ite, jts, jte, kts, kte
! temporary changed from inout to in

   real(c_float), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: &
                          qv, qc, qr, qi, qs, qg, ni, ns, nr, th, ng   
!jdf                      qndrop ! hm added, wrf-chem
   ! real, dimension(ims:ime, kms:kme, jms:jme), optional,intent(inout):: qndrop
   real(c_float), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: qndrop
!jdf  real, dimension(ims:ime, kms:kme, jms:jme),intent(inout):: csed3d, &
   ! real, dimension(ims:ime, kms:kme, jms:jme), optional,intent(inout):: qlsink, &
   !                        rainprod, evapprod, &
   !                        preci,precs,precg,precr ! hm, wrf-chem
   real(c_float), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: qlsink, &
                          rainprod, evapprod, &
                          preci,precs,precg,precr ! hm, wrf-chem
!, effcs, effis

   real(c_float), dimension(ims:ime, kms:kme, jms:jme), intent(in):: &
                          pii, p, dz, rho, w !, tke, nctend, nitend,kzh
   real(c_float), intent(in):: dt_in
   integer, intent(in):: itimestep

   real(c_float), dimension(ims:ime, jms:jme), intent(inout):: &
                          rainnc, rainncv, sr, &
! hm added 7/13/13
                          snownc,snowncv,graupelnc,graupelncv

   real(c_float), dimension(ims:ime, kms:kme, jms:jme), intent(inout)::       &  ! gt
                          refl_10cm

   ! logical, optional, intent(in) :: wetscav_on
   integer(c_int), intent(in) :: wetscav_on

   ! local variables

   real, dimension(its:ite, kts:kte, jts:jte)::                     &
                      effi, effs, effr, effg

   real, dimension(its:ite, kts:kte, jts:jte)::                     &
                      t, wvar, effc

   real, dimension(kts:kte) ::                                                                & 
                            qc_tend1d, qi_tend1d, qni_tend1d, qr_tend1d,                      &
                            ni_tend1d, ns_tend1d, nr_tend1d,                                  &
                            qc1d, qi1d, qr1d,ni1d, ns1d, nr1d, qs1d,                          &
                            t_tend1d,qv_tend1d, t1d, qv1d, p1d, w1d, wvar1d,         &
                            effc1d, effi1d, effs1d, effr1d,dz1d,   &
   ! hm add graupel
                            qg_tend1d, ng_tend1d, qg1d, ng1d, effg1d, &

! add sedimentation tendencies (units of kg/kg/s)
                            qgsten,qrsten, qisten, qnisten, qcsten, &
! add cumulus tendencies
                            qrcu1d, qscu1d, qicu1d

! add cumulus tendencies

   real, dimension(ims:ime, kms:kme, jms:jme), intent(in):: &
      qrcuten, qscuten, qicuten

  ! logical, intent(in), optional ::                f_qndrop  ! wrf-chem
  integer(c_int), intent(in) ::                f_qndrop  ! wrf-chem
  logical :: flag_qndrop  ! wrf-chem
  integer :: iinum ! wrf-chem

! wrf-chem
   real, dimension(kts:kte) :: nc1d, nc_tend1d,c2prec,csed,ised,ssed,gsed,rsed    
   real, dimension(kts:kte) :: rainprod1d, evapprod1d
! hm add reflectivity      
   real, dimension(kts:kte) :: dbz
                          
   real precprt1d, snowrt1d, snowprt1d, grplprt1d ! hm added 7/13/13

   integer i,k,j

   real dt

   ! logical, optional, intent(in) :: diagflag
   ! integer, optional, intent(in) :: do_radar_ref
   integer(c_int), intent(in) :: diagflag
   integer(c_int), intent(in) :: do_radar_ref

   logical :: has_wetscav

! below for wrf-chem
   flag_qndrop = .false.
   flag_qndrop = (f_qndrop == 1)
!!!!!!!!!!!!!!!!!!!!!!

   has_wetscav = (wetscav_on == 1)

   ! initialize tendencies (all set to 0) and transfer
   ! array to local variables
   dt = dt_in   

   do i=its,ite
   do j=jts,jte
   do k=kts,kte
       t(i,k,j)        = th(i,k,j)*pii(i,k,j)

! note: wvar not currently used in code !!!!!!!!!!
! currently assign wvar to 0.5 m/s (not coupled with pbl scheme)

       wvar(i,k,j)     = 0.5

! currently mixing of number concentrations also is neglected (not coupled with pbl schemes)

   end do
   end do
   end do

   do i=its,ite      ! i loop (east-west)
   do j=jts,jte      ! j loop (north-south)
   !
   ! transfer 3d arrays into 1d for microphysical calculations
   !

! hm , initialize 1d tendency arrays to zero

      do k=kts,kte   ! k loop (vertical)

          qc_tend1d(k)  = 0.
          qi_tend1d(k)  = 0.
          qni_tend1d(k) = 0.
          qr_tend1d(k)  = 0.
          ni_tend1d(k)  = 0.
          ns_tend1d(k)  = 0.
          nr_tend1d(k)  = 0.
          t_tend1d(k)   = 0.
          qv_tend1d(k)  = 0.
          nc_tend1d(k) = 0. ! wrf-chem

          qc1d(k)       = qc(i,k,j)
          qi1d(k)       = qi(i,k,j)
          qs1d(k)       = qs(i,k,j)
          qr1d(k)       = qr(i,k,j)

          ni1d(k)       = ni(i,k,j)

          ns1d(k)       = ns(i,k,j)
          nr1d(k)       = nr(i,k,j)
! hm add graupel
          qg1d(k)       = qg(i,k,j)
          ng1d(k)       = ng(i,k,j)
          qg_tend1d(k)  = 0.
          ng_tend1d(k)  = 0.

          t1d(k)        = t(i,k,j)
          qv1d(k)       = qv(i,k,j)
          p1d(k)        = p(i,k,j)
          dz1d(k)       = dz(i,k,j)
          w1d(k)        = w(i,k,j)
          wvar1d(k)     = wvar(i,k,j)
! add cumulus tendencies, already decoupled
          qrcu1d(k)     = qrcuten(i,k,j)
          qscu1d(k)     = qscuten(i,k,j)
          qicu1d(k)     = qicuten(i,k,j)
      end do  !jdf added this
! below for wrf-chem
   if (flag_qndrop) then
      iact = 3
      do k = kts, kte
         nc1d(k)=qndrop(i,k,j)
         iinum=0
      enddo
   else
      do k = kts, kte
         nc1d(k)=0. ! temporary placeholder, set to constant in microphysics subroutine
         iinum=1
      enddo
   endif

!jdf  end do

      call morr_two_moment_micro(qc_tend1d, qi_tend1d, qni_tend1d, qr_tend1d,            &
       ni_tend1d, ns_tend1d, nr_tend1d,                                                  &
       qc1d, qi1d, qs1d, qr1d,ni1d, ns1d, nr1d,                                          &
       t_tend1d,qv_tend1d, t1d, qv1d, p1d, dz1d, w1d, wvar1d,                   &
       precprt1d,snowrt1d,                                                               &
       snowprt1d,grplprt1d,                 & ! hm added 7/13/13
       effc1d,effi1d,effs1d,effr1d,dt,                                                   &
                                            ims,ime, jms,jme, kms,kme,                   &
                                            its,ite, jts,jte, kts,kte,                   & ! hm add graupel
                                    qg_tend1d,ng_tend1d,qg1d,ng1d,effg1d, &
                                    qrcu1d, qscu1d, qicu1d, &
! add sedimentation tendencies
                                  qgsten,qrsten,qisten,qnisten,qcsten, &
                                  nc1d, nc_tend1d, iinum, c2prec,csed,ised,ssed,gsed,rsed & !wrf-chem
#if (wrf_chem == 1)
                                  ,has_wetscav,rainprod1d, evapprod1d & !wrf-chem
#endif
                       )

   !
   ! transfer 1d arrays back into 3d arrays
   !
      do k=kts,kte

! hm, add tendencies to update global variables 
! hm, tendencies for q and n now added in m2005micro, so we
! only need to transfer 1d variables back to 3d

          qc(i,k,j)        = qc1d(k)
          qi(i,k,j)        = qi1d(k)
          qs(i,k,j)        = qs1d(k)
          qr(i,k,j)        = qr1d(k)
          ni(i,k,j)        = ni1d(k)
          ns(i,k,j)        = ns1d(k)          
          nr(i,k,j)        = nr1d(k)
	  qg(i,k,j)        = qg1d(k)
          ng(i,k,j)        = ng1d(k)

          t(i,k,j)         = t1d(k)
          th(i,k,j)        = t(i,k,j)/pii(i,k,j) ! convert temp back to potential temp
          qv(i,k,j)        = qv1d(k)

          effc(i,k,j)      = effc1d(k)
          effi(i,k,j)      = effi1d(k)
          effs(i,k,j)      = effs1d(k)
          effr(i,k,j)      = effr1d(k)
	  effg(i,k,j)      = effg1d(k)

! wrf-chem
          if (flag_qndrop) then
             qndrop(i,k,j) = nc1d(k)
!jdf         csed3d(i,k,j) = csed(k)
          end if
           if(qc(i,k,j)>1.e-10) then
              qlsink(i,k,j)  = c2prec(k)/qc(i,k,j)
           else
              qlsink(i,k,j)  = 0.0
           endif
          precr(i,k,j) = rsed(k)
          preci(i,k,j) = ised(k)
          precs(i,k,j) = ssed(k)
          precg(i,k,j) = gsed(k)
! effective radius for radiation code (currently not coupled)
! hm, add limit to prevent blowing up optical properties, 8/18/07
!          effcs(i,k,j)     = min(effc(i,k,j),50.)
!          effcs(i,k,j)     = max(effcs(i,k,j),1.)
!          effis(i,k,j)     = min(effi(i,k,j),130.)
!          effis(i,k,j)     = max(effis(i,k,j),13.)

#if ( wrf_chem == 1)
           if ( has_wetscav ) then
             rainprod(i,k,j) = rainprod1d(k)
             evapprod(i,k,j) = evapprod1d(k)
           endif
#endif

      end do

! hm modified so that m2005 precip variables correctly match wrf precip variables
      rainnc(i,j) = rainnc(i,j)+precprt1d
      rainncv(i,j) = precprt1d
! hm, added 7/13/13
      snownc(i,j) = snownc(i,j)+snowprt1d
      snowncv(i,j) = snowprt1d
      graupelnc(i,j) = graupelnc(i,j)+grplprt1d
      graupelncv(i,j) = grplprt1d
      sr(i,j) = snowrt1d/(precprt1d+1.e-12)

!+---+-----------------------------------------------------------------+
         if ((diagflag == 1) .and. do_radar_ref == 1) then
          call refl10cm_hm (qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d,   &
                      t1d, p1d, dbz, kts, kte, i, j)
          do k = kts, kte
             refl_10cm(i,k,j) = max(-35., dbz(k))
          enddo
         endif
!+---+-----------------------------------------------------------------+

   end do
   end do   

end subroutine mp_morr_two_moment

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine morr_two_moment_micro(qc3dten,qi3dten,qni3dten,qr3dten,         &
       ni3dten,ns3dten,nr3dten,qc3d,qi3d,qni3d,qr3d,ni3d,ns3d,nr3d,              &
       t3dten,qv3dten,t3d,qv3d,pres,dzq,w3d,wvar,precrt,snowrt,            &
       snowprt,grplprt,                & ! hm added 7/13/13
       effc,effi,effs,effr,dt,                                                   &
                                            ims,ime, jms,jme, kms,kme,           &
                                            its,ite, jts,jte, kts,kte,           & ! add graupel
                        qg3dten,ng3dten,qg3d,ng3d,effg,qrcu1d,qscu1d, qicu1d,    &
                        qgsten,qrsten,qisten,qnisten,qcsten, &
                        nc3d,nc3dten,iinum, & ! wrf-chem
				c2prec,csed,ised,ssed,gsed,rsed  &  ! hm added, wrf-chem
#if (wrf_chem == 1)
        ,has_wetscav,rainprod, evapprod &
#endif
                        )

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! this program is the main two-moment microphysics subroutine described by
! morrison et al. 2005 jas and morrison et al. 2009 mwr

! this scheme is a bulk double-moment scheme that predicts mixing
! ratios and number concentrations of five hydrometeor species:
! cloud droplets, cloud (small) ice, rain, snow, and graupel/hail.

! code structure: main subroutine is 'morr_two_moment'. also included in this file is
! 'function polysvp', 'function derf1', and
! 'function gamma'.

! note: this subroutine uses 1d array in vertical (column), even though variables are called '3d'......

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

! declarations

      implicit none

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! these variables below must be linked with the main model.
! define array sizes

! input number of grid cells

! input/output parameters                                 ! description (units)
      integer, intent( in)  :: ims,ime, jms,jme, kms,kme,          &
                               its,ite, jts,jte, kts,kte

      real, dimension(kts:kte) ::  qc3dten            ! cloud water mixing ratio tendency (kg/kg/s)
      real, dimension(kts:kte) ::  qi3dten            ! cloud ice mixing ratio tendency (kg/kg/s)
      real, dimension(kts:kte) ::  qni3dten           ! snow mixing ratio tendency (kg/kg/s)
      real, dimension(kts:kte) ::  qr3dten            ! rain mixing ratio tendency (kg/kg/s)
      real, dimension(kts:kte) ::  ni3dten            ! cloud ice number concentration (1/kg/s)
      real, dimension(kts:kte) ::  ns3dten            ! snow number concentration (1/kg/s)
      real, dimension(kts:kte) ::  nr3dten            ! rain number concentration (1/kg/s)
      real, dimension(kts:kte) ::  qc3d               ! cloud water mixing ratio (kg/kg)
      real, dimension(kts:kte) ::  qi3d               ! cloud ice mixing ratio (kg/kg)
      real, dimension(kts:kte) ::  qni3d              ! snow mixing ratio (kg/kg)
      real, dimension(kts:kte) ::  qr3d               ! rain mixing ratio (kg/kg)
      real, dimension(kts:kte) ::  ni3d               ! cloud ice number concentration (1/kg)
      real, dimension(kts:kte) ::  ns3d               ! snow number concentration (1/kg)
      real, dimension(kts:kte) ::  nr3d               ! rain number concentration (1/kg)
      real, dimension(kts:kte) ::  t3dten             ! temperature tendency (k/s)
      real, dimension(kts:kte) ::  qv3dten            ! water vapor mixing ratio tendency (kg/kg/s)
      real, dimension(kts:kte) ::  t3d                ! temperature (k)
      real, dimension(kts:kte) ::  qv3d               ! water vapor mixing ratio (kg/kg)
      real, dimension(kts:kte) ::  pres               ! atmospheric pressure (pa)
      real, dimension(kts:kte) ::  dzq                ! difference in height across level (m)
      real, dimension(kts:kte) ::  w3d                ! grid-scale vertical velocity (m/s)
      real, dimension(kts:kte) ::  wvar               ! sub-grid vertical velocity (m/s)
! below for wrf-chem
      real, dimension(kts:kte) ::  nc3d
      real, dimension(kts:kte) ::  nc3dten
      integer, intent(in) :: iinum

! hm added graupel variables
      real, dimension(kts:kte) ::  qg3dten            ! graupel mix ratio tendency (kg/kg/s)
      real, dimension(kts:kte) ::  ng3dten            ! graupel numb conc tendency (1/kg/s)
      real, dimension(kts:kte) ::  qg3d            ! graupel mix ratio (kg/kg)
      real, dimension(kts:kte) ::  ng3d            ! graupel number conc (1/kg)

! hm, add 1/16/07, sedimentation tendencies for mixing ratio

      real, dimension(kts:kte) ::  qgsten            ! graupel sed tend (kg/kg/s)
      real, dimension(kts:kte) ::  qrsten            ! rain sed tend (kg/kg/s)
      real, dimension(kts:kte) ::  qisten            ! cloud ice sed tend (kg/kg/s)
      real, dimension(kts:kte) ::  qnisten           ! snow sed tend (kg/kg/s)
      real, dimension(kts:kte) ::  qcsten            ! cloud wat sed tend (kg/kg/s)      

! hm add cumulus tendencies for precip
        real, dimension(kts:kte) ::   qrcu1d
        real, dimension(kts:kte) ::   qscu1d
        real, dimension(kts:kte) ::   qicu1d

! output variables

        real precrt                ! total precip per time step (mm)
        real snowrt                ! snow per time step (mm)
! hm added 7/13/13
        real snowprt      ! total cloud ice plus snow per time step (mm)
	real grplprt	  ! total graupel per time step (mm)

        real, dimension(kts:kte) ::   effc            ! droplet effective radius (micron)
        real, dimension(kts:kte) ::   effi            ! cloud ice effective radius (micron)
        real, dimension(kts:kte) ::   effs            ! snow effective radius (micron)
        real, dimension(kts:kte) ::   effr            ! rain effective radius (micron)
        real, dimension(kts:kte) ::   effg            ! graupel effective radius (micron)

! model input parameters (formerly in common blocks)

        real dt         ! model time step (sec)

#if (wrf_chem == 1)
      logical, intent(in) :: has_wetscav
#endif


!.....................................................................................................
! local variables: all parameters below are local to scheme and don't need to communicate with the
! rest of the model.

! size parameter variables

     real, dimension(kts:kte) :: lamc          ! slope parameter for droplets (m-1)
     real, dimension(kts:kte) :: lami          ! slope parameter for cloud ice (m-1)
     real, dimension(kts:kte) :: lams          ! slope parameter for snow (m-1)
     real, dimension(kts:kte) :: lamr          ! slope parameter for rain (m-1)
     real, dimension(kts:kte) :: lamg          ! slope parameter for graupel (m-1)
     real, dimension(kts:kte) :: cdist1        ! psd parameter for droplets
     real, dimension(kts:kte) :: n0i           ! intercept parameter for cloud ice (kg-1 m-1)
     real, dimension(kts:kte) :: n0s           ! intercept parameter for snow (kg-1 m-1)
     real, dimension(kts:kte) :: n0rr          ! intercept parameter for rain (kg-1 m-1)
     real, dimension(kts:kte) :: n0g           ! intercept parameter for graupel (kg-1 m-1)
     real, dimension(kts:kte) :: pgam          ! spectral shape parameter for droplets

! microphysical processes

     real, dimension(kts:kte) ::  nsubc     ! loss of nc during evap
     real, dimension(kts:kte) ::  nsubi     ! loss of ni during sub.
     real, dimension(kts:kte) ::  nsubs     ! loss of ns during sub.
     real, dimension(kts:kte) ::  nsubr     ! loss of nr during evap
     real, dimension(kts:kte) ::  prd       ! dep cloud ice
     real, dimension(kts:kte) ::  pre       ! evap of rain
     real, dimension(kts:kte) ::  prds      ! dep snow
     real, dimension(kts:kte) ::  nnuccc    ! change n due to contact freez droplets
     real, dimension(kts:kte) ::  mnuccc    ! change q due to contact freez droplets
     real, dimension(kts:kte) ::  pra       ! accretion droplets by rain
     real, dimension(kts:kte) ::  prc       ! autoconversion droplets
     real, dimension(kts:kte) ::  pcc       ! cond/evap droplets
     real, dimension(kts:kte) ::  nnuccd    ! change n freezing aerosol (prim ice nucleation)
     real, dimension(kts:kte) ::  mnuccd    ! change q freezing aerosol (prim ice nucleation)
     real, dimension(kts:kte) ::  mnuccr    ! change q due to contact freez rain
     real, dimension(kts:kte) ::  nnuccr    ! change n due to contact freez rain
     real, dimension(kts:kte) ::  npra      ! change in n due to droplet acc by rain
     real, dimension(kts:kte) ::  nragg     ! self-collection/breakup of rain
     real, dimension(kts:kte) ::  nsagg     ! self-collection of snow
     real, dimension(kts:kte) ::  nprc      ! change nc autoconversion droplets
     real, dimension(kts:kte) ::  nprc1      ! change nr autoconversion droplets
     real, dimension(kts:kte) ::  prai      ! change q accretion cloud ice by snow
     real, dimension(kts:kte) ::  prci      ! change q autoconversin cloud ice to snow
     real, dimension(kts:kte) ::  psacws    ! change q droplet accretion by snow
     real, dimension(kts:kte) ::  npsacws   ! change n droplet accretion by snow
     real, dimension(kts:kte) ::  psacwi    ! change q droplet accretion by cloud ice
     real, dimension(kts:kte) ::  npsacwi   ! change n droplet accretion by cloud ice
     real, dimension(kts:kte) ::  nprci     ! change n autoconversion cloud ice by snow
     real, dimension(kts:kte) ::  nprai     ! change n accretion cloud ice
     real, dimension(kts:kte) ::  nmults    ! ice mult due to riming droplets by snow
     real, dimension(kts:kte) ::  nmultr    ! ice mult due to riming rain by snow
     real, dimension(kts:kte) ::  qmults    ! change q due to ice mult droplets/snow
     real, dimension(kts:kte) ::  qmultr    ! change q due to ice rain/snow
     real, dimension(kts:kte) ::  pracs     ! change q rain-snow collection
     real, dimension(kts:kte) ::  npracs    ! change n rain-snow collection
     real, dimension(kts:kte) ::  pccn      ! change q droplet activation
     real, dimension(kts:kte) ::  psmlt     ! change q melting snow to rain
     real, dimension(kts:kte) ::  evpms     ! chnage q melting snow evaporating
     real, dimension(kts:kte) ::  nsmlts    ! change n melting snow
     real, dimension(kts:kte) ::  nsmltr    ! change n melting snow to rain
! hm added 12/13/06
     real, dimension(kts:kte) ::  piacr     ! change qr, ice-rain collection
     real, dimension(kts:kte) ::  niacr     ! change n, ice-rain collection
     real, dimension(kts:kte) ::  praci     ! change qi, ice-rain collection
     real, dimension(kts:kte) ::  piacrs     ! change qr, ice rain collision, added to snow
     real, dimension(kts:kte) ::  niacrs     ! change n, ice rain collision, added to snow
     real, dimension(kts:kte) ::  pracis     ! change qi, ice rain collision, added to snow
     real, dimension(kts:kte) ::  eprd      ! sublimation cloud ice
     real, dimension(kts:kte) ::  eprds     ! sublimation snow
! hm added graupel processes
     real, dimension(kts:kte) ::  pracg    ! change in q collection rain by graupel
     real, dimension(kts:kte) ::  psacwg    ! change in q collection droplets by graupel
     real, dimension(kts:kte) ::  pgsacw    ! conversion q to graupel due to collection droplets by snow
     real, dimension(kts:kte) ::  pgracs    ! conversion q to graupel due to collection rain by snow
     real, dimension(kts:kte) ::  prdg    ! dep of graupel
     real, dimension(kts:kte) ::  eprdg    ! sub of graupel
     real, dimension(kts:kte) ::  evpmg    ! change q melting of graupel and evaporation
     real, dimension(kts:kte) ::  pgmlt    ! change q melting of graupel
     real, dimension(kts:kte) ::  npracg    ! change n collection rain by graupel
     real, dimension(kts:kte) ::  npsacwg    ! change n collection droplets by graupel
     real, dimension(kts:kte) ::  nscng    ! change n conversion to graupel due to collection droplets by snow
     real, dimension(kts:kte) ::  ngracs    ! change n conversion to graupel due to collection rain by snow
     real, dimension(kts:kte) ::  ngmltg    ! change n melting graupel
     real, dimension(kts:kte) ::  ngmltr    ! change n melting graupel to rain
     real, dimension(kts:kte) ::  nsubg    ! change n sub/dep of graupel
     real, dimension(kts:kte) ::  psacr    ! conversion due to coll of snow by rain
     real, dimension(kts:kte) ::  nmultg    ! ice mult due to acc droplets by graupel
     real, dimension(kts:kte) ::  nmultrg    ! ice mult due to acc rain by graupel
     real, dimension(kts:kte) ::  qmultg    ! change q due to ice mult droplets/graupel
     real, dimension(kts:kte) ::  qmultrg    ! change q due to ice mult rain/graupel

! time-varying atmospheric parameters

     real, dimension(kts:kte) ::   kap   ! thermal conductivity of air
     real, dimension(kts:kte) ::   evs   ! saturation vapor pressure
     real, dimension(kts:kte) ::   eis   ! ice saturation vapor pressure
     real, dimension(kts:kte) ::   qvs   ! saturation mixing ratio
     real, dimension(kts:kte) ::   qvi   ! ice saturation mixing ratio
     real, dimension(kts:kte) ::   qvqvs ! sautration ratio
     real, dimension(kts:kte) ::   qvqvsi! ice saturaion ratio
     real, dimension(kts:kte) ::   dv    ! diffusivity of water vapor in air
     real, dimension(kts:kte) ::   xxls  ! latent heat of sublimation
     real, dimension(kts:kte) ::   xxlv  ! latent heat of vaporization
     real, dimension(kts:kte) ::   cpm   ! specific heat at const pressure for moist air
     real, dimension(kts:kte) ::   mu    ! viscocity of air
     real, dimension(kts:kte) ::   sc    ! schmidt number
     real, dimension(kts:kte) ::   xlf   ! latent heat of freezing
     real, dimension(kts:kte) ::   rho   ! air density
     real, dimension(kts:kte) ::   ab    ! correction to condensation rate due to latent heating
     real, dimension(kts:kte) ::   abi    ! correction to deposition rate due to latent heating

! time-varying microphysics parameters

     real, dimension(kts:kte) ::   dap    ! diffusivity of aerosol
     real    nacnt                    ! number of contact in
     real    fmult                    ! temp.-dep. parameter for rime-splintering
     real    coffi                    ! ice autoconversion parameter

! fall speed working variables (defined in code)

      real, dimension(kts:kte) ::    dumi,dumr,dumfni,dumg,dumfng
      real uni, umi,umr
      real, dimension(kts:kte) ::    fr, fi, fni,fg,fng
      real rgvm
      real, dimension(kts:kte) ::   faloutr,falouti,faloutni
      real faltndr,faltndi,faltndni,rho2
      real, dimension(kts:kte) ::   dumqs,dumfns
      real ums,uns
      real, dimension(kts:kte) ::   fs,fns, falouts,faloutns,faloutg,faloutng
      real faltnds,faltndns,unr,faltndg,faltndng
      real, dimension(kts:kte) ::    dumc,dumfnc
      real unc,umc,ung,umg
      real, dimension(kts:kte) ::   fc,faloutc,faloutnc
      real faltndc,faltndnc
      real, dimension(kts:kte) ::   fnc,dumfnr,faloutnr
      real faltndnr
      real, dimension(kts:kte) ::   fnr

! fall-speed parameter 'a' with air density correction

      real, dimension(kts:kte) ::    ain,arn,asn,acn,agn

! external function call return variables

!      real gamma,      ! euler gamma function
!      real polysvp,    ! sat. pressure function
!      real derf1        ! error function

! dummy variables

     real dum,dum1,dum2,dumt,dumqv,dumqss,dumqsi,dums

! prognostic supersaturation

     real dqsdt    ! change of sat. mix. rat. with temperature
     real dqsidt   ! change in ice sat. mixing rat. with t
     real epsi     ! 1/phase rel. time (see m2005), ice
     real epss     ! 1/phase rel. time (see m2005), snow
     real epsr     ! 1/phase rel. time (see m2005), rain
     real epsg     ! 1/phase rel. time (see m2005), graupel

! new droplet activation variables
     real tauc     ! phase rel. time (see m2005), droplets
     real taur     ! phase rel. time (see m2005), rain
     real taui     ! phase rel. time (see m2005), cloud ice
     real taus     ! phase rel. time (see m2005), snow
     real taug     ! phase rel. time (see m2005), graupel
     real dumact,dum3

! counting/index variables

     integer k,nstep,n ! ,i

! ltrue is only used to speed up the code !!
! ltrue, switch = 0, no hydrometeors in column, 
!               = 1, hydrometeors in column

      integer ltrue

! droplet activation/freezing aerosol


     real    ct      ! droplet activation parameter
     real    temp1   ! dummy temperature
     real    sat1    ! dummy saturation
     real    sigvl   ! surface tension liq/vapor
     real    kel     ! kelvin parameter
     real    kc2     ! total ice nucleation rate

       real cry,kry   ! aerosol activation parameters

! more working/dummy variables

     real dumqi,dumni,dc0,ds0,dg0
     real dumqc,dumqr,ratio,sum_dep,fudgef

! effective vertical velocity  (m/s)
     real wef

! working parameters for ice nucleation

      real anuc,bnuc

! working parameters for aerosol activation

        real aact,gamm,gg,psi,eta1,eta2,sm1,sm2,smax,uu1,uu2,alpha

! dummy size distribution parameters

        real dlams,dlamr,dlami,dlamc,dlamg,lammax,lammin

        integer idrop

! for wrf-chem
	real, dimension(kts:kte)::c2prec,csed,ised,ssed,gsed,rsed
#if (wrf_chem == 1)
    real, dimension(kts:kte), intent(inout) :: rainprod, evapprod
#endif
    real, dimension(kts:kte)                :: tqimelt ! melting of cloud ice (tendency)

! comment lines for wrf-chem since these are intent(in) in that case
!       real, dimension(kts:kte) ::  nc3dten            ! cloud droplet number concentration (1/kg/s)
!       real, dimension(kts:kte) ::  nc3d               ! cloud droplet number concentration (1/kg)

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

! set ltrue initially to 0

         ltrue = 0

! atmospheric parameters that vary in time and height
         do k = kts,kte

! nc3dten local array initialized
               nc3dten(k) = 0.
! initialize variables for wrf-chem output to zero

		c2prec(k)=0.
		csed(k)=0.
		ised(k)=0.
		ssed(k)=0.
		gsed(k)=0.
		rsed(k)=0.

#if (wrf_chem == 1)
         rainprod(k) = 0.
         evapprod(k) = 0.
         tqimelt(k)  = 0.
         prc(k)      = 0.
         pra(k)      = 0.
#endif

! latent heat of vaporation

            xxlv(k) = 3.1484e6-2370.*t3d(k)

! latent heat of sublimation

            xxls(k) = 3.15e6-2370.*t3d(k)+0.3337e6

            cpm(k) = cp*(1.+0.887*qv3d(k))

! saturation vapor pressure and mixing ratio

! hm, add fix for low pressure, 5/12/10

            evs(k) = min(0.99*pres(k),polysvp(t3d(k),0))   ! pa
            eis(k) = min(0.99*pres(k),polysvp(t3d(k),1))   ! pa

! make sure ice saturation doesn't exceed water sat. near freezing

            if (eis(k).gt.evs(k)) eis(k) = evs(k)

            qvs(k) = ep_2*evs(k)/(pres(k)-evs(k))
            qvi(k) = ep_2*eis(k)/(pres(k)-eis(k))

            qvqvs(k) = qv3d(k)/qvs(k)
            qvqvsi(k) = qv3d(k)/qvi(k)

! air density

            rho(k) = pres(k)/(r*t3d(k))

! add number concentration due to cumulus tendency
! assume n0 associated with cumulus param rain is 10^7 m^-4
! assume n0 associated with cumulus param snow is 2 x 10^7 m^-4
! for detrained cloud ice, assume mean volume diam of 80 micron

            if (qrcu1d(k).ge.1.e-10) then
            dum=1.8e5*(qrcu1d(k)*dt/(pi*rhow*rho(k)**3))**0.25
            nr3d(k)=nr3d(k)+dum
            end if
            if (qscu1d(k).ge.1.e-10) then
            dum=3.e5*(qscu1d(k)*dt/(cons1*rho(k)**3))**(1./(ds+1.))
            ns3d(k)=ns3d(k)+dum
            end if
            if (qicu1d(k).ge.1.e-10) then
            dum=qicu1d(k)*dt/(ci*(80.e-6)**di)
            ni3d(k)=ni3d(k)+dum
            end if

! at subsaturation, remove small amounts of cloud/precip water
! hm modify 7/0/09 change limit to 1.e-8

             if (qvqvs(k).lt.0.9) then
               if (qr3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qr3d(k)
                  t3d(k)=t3d(k)-qr3d(k)*xxlv(k)/cpm(k)
                  qr3d(k)=0.
               end if
               if (qc3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qc3d(k)
                  t3d(k)=t3d(k)-qc3d(k)*xxlv(k)/cpm(k)
                  qc3d(k)=0.
               end if
             end if

             if (qvqvsi(k).lt.0.9) then
               if (qi3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qi3d(k)
                  t3d(k)=t3d(k)-qi3d(k)*xxls(k)/cpm(k)
                  qi3d(k)=0.
               end if
               if (qni3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qni3d(k)
                  t3d(k)=t3d(k)-qni3d(k)*xxls(k)/cpm(k)
                  qni3d(k)=0.
               end if
               if (qg3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qg3d(k)
                  t3d(k)=t3d(k)-qg3d(k)*xxls(k)/cpm(k)
                  qg3d(k)=0.
               end if
             end if

! heat of fusion

            xlf(k) = xxls(k)-xxlv(k)

!..................................................................
! if mixing ratio < qsmall set mixing ratio and number conc to zero

       if (qc3d(k).lt.qsmall) then
         qc3d(k) = 0.
         nc3d(k) = 0.
         effc(k) = 0.
       end if
       if (qr3d(k).lt.qsmall) then
         qr3d(k) = 0.
         nr3d(k) = 0.
         effr(k) = 0.
       end if
       if (qi3d(k).lt.qsmall) then
         qi3d(k) = 0.
         ni3d(k) = 0.
         effi(k) = 0.
       end if
       if (qni3d(k).lt.qsmall) then
         qni3d(k) = 0.
         ns3d(k) = 0.
         effs(k) = 0.
       end if
       if (qg3d(k).lt.qsmall) then
         qg3d(k) = 0.
         ng3d(k) = 0.
         effg(k) = 0.
       end if

! initialize sedimentation tendencies for mixing ratio

      qrsten(k) = 0.
      qisten(k) = 0.
      qnisten(k) = 0.
      qcsten(k) = 0.
      qgsten(k) = 0.

!..................................................................
! microphysics parameters varying in time/height

! fix 053011
            mu(k) = 1.496e-6*t3d(k)**1.5/(t3d(k)+120.)

! fall speed with density correction (heymsfield and benssemer 2006)

            dum = (rhosu/rho(k))**0.54

! fix 053011
!            ain(k) = dum*ai
! aa revision 4/1/11: ikawa and saito 1991 air-density correction 
            ain(k) = (rhosu/rho(k))**0.35*ai
            arn(k) = dum*ar
            asn(k) = dum*as
!            acn(k) = dum*ac
! aa revision 4/1/11: temperature-dependent stokes fall speed
            acn(k) = g*rhow/(18.*mu(k))
! hm add graupel 8/28/06
            agn(k) = dum*ag

!hm 4/7/09 bug fix, initialize lami to prevent later division by zero
            lami(k)=0.

!..................................
! if there is no cloud/precip water, and if subsaturated, then skip microphysics
! for this level

            if (qc3d(k).lt.qsmall.and.qi3d(k).lt.qsmall.and.qni3d(k).lt.qsmall &
                 .and.qr3d(k).lt.qsmall.and.qg3d(k).lt.qsmall) then
                 if (t3d(k).lt.273.15.and.qvqvsi(k).lt.0.999) goto 200
                 if (t3d(k).ge.273.15.and.qvqvs(k).lt.0.999) goto 200
            end if

! thermal conductivity for air

! fix 053011
            kap(k) = 1.414e3*mu(k)

! diffusivity of water vapor

            dv(k) = 8.794e-5*t3d(k)**1.81/pres(k)

! schmit number

! fix 053011
            sc(k) = mu(k)/(rho(k)*dv(k))

! psychometic corrections

! rate of change sat. mix. ratio with temperature

            dum = (rv*t3d(k)**2)

            dqsdt = xxlv(k)*qvs(k)/dum
            dqsidt =  xxls(k)*qvi(k)/dum

            abi(k) = 1.+dqsidt*xxls(k)/cpm(k)
            ab(k) = 1.+dqsdt*xxlv(k)/cpm(k)

! 
!.....................................................................
!.....................................................................
! case for temperature above freezing

            if (t3d(k).ge.273.15) then

!......................................................................
!hm add, allow for constant droplet number
! inum = 0, predict droplet number
! inum = 1, set constant droplet number

         if (iinum.eq.1) then
! convert ndcnst from cm-3 to kg-1
            nc3d(k)=ndcnst*1.e6/rho(k)
         end if

! get size distribution parameters

! melt very small snow and graupel mixing ratios, add to rain
       if (qni3d(k).lt.1.e-6) then
          qr3d(k)=qr3d(k)+qni3d(k)
          nr3d(k)=nr3d(k)+ns3d(k)
          t3d(k)=t3d(k)-qni3d(k)*xlf(k)/cpm(k)
          qni3d(k) = 0.
          ns3d(k) = 0.
       end if
       if (qg3d(k).lt.1.e-6) then
          qr3d(k)=qr3d(k)+qg3d(k)
          nr3d(k)=nr3d(k)+ng3d(k)
          t3d(k)=t3d(k)-qg3d(k)*xlf(k)/cpm(k)
          qg3d(k) = 0.
          ng3d(k) = 0.
       end if

       if (qc3d(k).lt.qsmall.and.qni3d(k).lt.1.e-8.and.qr3d(k).lt.qsmall.and.qg3d(k).lt.1.e-8) goto 300

! make sure number concentrations aren't negative

      ns3d(k) = max(0.,ns3d(k))
      nc3d(k) = max(0.,nc3d(k))
      nr3d(k) = max(0.,nr3d(k))
      ng3d(k) = max(0.,ng3d(k))

!......................................................................
! rain

      if (qr3d(k).ge.qsmall) then
      lamr(k) = (pi*rhow*nr3d(k)/qr3d(k))**(1./3.)
      n0rr(k) = nr3d(k)*lamr(k)

! check for slope

! adjust vars

      if (lamr(k).lt.lamminr) then

      lamr(k) = lamminr

      n0rr(k) = lamr(k)**4*qr3d(k)/(pi*rhow)

      nr3d(k) = n0rr(k)/lamr(k)
      else if (lamr(k).gt.lammaxr) then
      lamr(k) = lammaxr
      n0rr(k) = lamr(k)**4*qr3d(k)/(pi*rhow)

      nr3d(k) = n0rr(k)/lamr(k)
      end if
      end if

!......................................................................
! cloud droplets

! martin et al. (1994) formula for pgam

      if (qc3d(k).ge.qsmall) then

         dum = pres(k)/(287.15*t3d(k))
         pgam(k)=0.0005714*(nc3d(k)/1.e6*dum)+0.2714
         pgam(k)=1./(pgam(k)**2)-1.
         pgam(k)=max(pgam(k),2.)
         pgam(k)=min(pgam(k),10.)

! calculate lamc

      lamc(k) = (cons26*nc3d(k)*gamma(pgam(k)+4.)/   &
                 (qc3d(k)*gamma(pgam(k)+1.)))**(1./3.)

! lammin, 60 micron diameter
! lammax, 1 micron

      lammin = (pgam(k)+1.)/60.e-6
      lammax = (pgam(k)+1.)/1.e-6

      if (lamc(k).lt.lammin) then
      lamc(k) = lammin

      nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+              &
                log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26
      else if (lamc(k).gt.lammax) then
      lamc(k) = lammax

      nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+              &
                log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26

      end if

      end if

!......................................................................
! snow

      if (qni3d(k).ge.qsmall) then
      lams(k) = (cons1*ns3d(k)/qni3d(k))**(1./ds)
      n0s(k) = ns3d(k)*lams(k)

! check for slope

! adjust vars

      if (lams(k).lt.lammins) then
      lams(k) = lammins
      n0s(k) = lams(k)**4*qni3d(k)/cons1

      ns3d(k) = n0s(k)/lams(k)

      else if (lams(k).gt.lammaxs) then

      lams(k) = lammaxs
      n0s(k) = lams(k)**4*qni3d(k)/cons1

      ns3d(k) = n0s(k)/lams(k)
      end if
      end if

!......................................................................
! graupel

      if (qg3d(k).ge.qsmall) then
      lamg(k) = (cons2*ng3d(k)/qg3d(k))**(1./dg)
      n0g(k) = ng3d(k)*lamg(k)

! adjust vars

      if (lamg(k).lt.lamming) then
      lamg(k) = lamming
      n0g(k) = lamg(k)**4*qg3d(k)/cons2

      ng3d(k) = n0g(k)/lamg(k)

      else if (lamg(k).gt.lammaxg) then

      lamg(k) = lammaxg
      n0g(k) = lamg(k)**4*qg3d(k)/cons2

      ng3d(k) = n0g(k)/lamg(k)
      end if
      end if

!.....................................................................
! zero out process rates

            prc(k) = 0.
            nprc(k) = 0.
            nprc1(k) = 0.
            pra(k) = 0.
            npra(k) = 0.
            nragg(k) = 0.
            nsmlts(k) = 0.
            nsmltr(k) = 0.
            evpms(k) = 0.
            pcc(k) = 0.
            pre(k) = 0.
            nsubc(k) = 0.
            nsubr(k) = 0.
            pracg(k) = 0.
            npracg(k) = 0.
            psmlt(k) = 0.
            pgmlt(k) = 0.
            evpmg(k) = 0.
            pracs(k) = 0.
            npracs(k) = 0.
            ngmltg(k) = 0.
            ngmltr(k) = 0.

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! calculation of microphysical process rates, t > 273.15 k

!.................................................................
!.......................................................................
! autoconversion of cloud liquid water to rain
! formula from beheng (1994)
! using numerical simulation of stochastic collection equation
! and initial cloud droplet size distribution specified
! as a gamma distribution

! use minimum value of 1.e-6 to prevent floating point error

         if (qc3d(k).ge.1.e-6) then

! hm add 12/13/06, replace with newer formula
! from khairoutdinov and kogan 2000, mwr

                prc(k)=1350.*qc3d(k)**2.47*  &
           (nc3d(k)/1.e6*rho(k))**(-1.79)

! note: nprc1 is change in nr,
! nprc is change in nc

        nprc1(k) = prc(k)/cons29
        nprc(k) = prc(k)/(qc3d(k)/nc3d(k))

! hm bug fix 3/20/12
                nprc(k) = min(nprc(k),nc3d(k)/dt)
                nprc1(k) = min(nprc1(k),nprc(k))

         end if

!.......................................................................
! hm add 12/13/06, collection of snow by rain above freezing
! formula from ikawa and saito (1991)

         if (qr3d(k).ge.1.e-8.and.qni3d(k).ge.1.e-8) then

            ums = asn(k)*cons3/(lams(k)**bs)
            umr = arn(k)*cons4/(lamr(k)**br)
            uns = asn(k)*cons5/lams(k)**bs
            unr = arn(k)*cons6/lamr(k)**br

! set reaslistic limits on fallspeeds

! bug fix, 10/08/09
            dum=(rhosu/rho(k))**0.54
            ums=min(ums,1.2*dum)
            uns=min(uns,1.2*dum)
            umr=min(umr,9.1*dum)
            unr=min(unr,9.1*dum)

! hm fix, 2/12/13
! for above freezing conditions to get accelerated melting of snow,
! we need collection of rain by snow (following lin et al. 1983)
!            pracs(k) = cons31*(((1.2*umr-0.95*ums)**2+              &
!                  0.08*ums*umr)**0.5*rho(k)*                     &
!                 n0rr(k)*n0s(k)/lams(k)**3*                    &
!                  (5./(lams(k)**3*lamr(k))+                    &
!                  2./(lams(k)**2*lamr(k)**2)+                  &
!                  0.5/(lams(k)*lamr(k)**3)))

            pracs(k) = cons41*(((1.2*umr-0.95*ums)**2+                   &
                  0.08*ums*umr)**0.5*rho(k)*                      &
                  n0rr(k)*n0s(k)/lamr(k)**3*                              &
                  (5./(lamr(k)**3*lams(k))+                    &
                  2./(lamr(k)**2*lams(k)**2)+                  &				 
                  0.5/(lamr(k)*lams(k)**3)))

! fix 053011, npracs no longer subtracted from snow
!            npracs(k) = cons32*rho(k)*(1.7*(unr-uns)**2+            &
!                0.3*unr*uns)**0.5*n0rr(k)*n0s(k)*              &
!                (1./(lamr(k)**3*lams(k))+                      &
!                 1./(lamr(k)**2*lams(k)**2)+                   &
!                 1./(lamr(k)*lams(k)**3))

         end if

! add collection of graupel by rain above freezing
! assume all rain collection by graupel above freezing is shed
! assume shed drops are 1 mm in size

         if (qr3d(k).ge.1.e-8.and.qg3d(k).ge.1.e-8) then

            umg = agn(k)*cons7/(lamg(k)**bg)
            umr = arn(k)*cons4/(lamr(k)**br)
            ung = agn(k)*cons8/lamg(k)**bg
            unr = arn(k)*cons6/lamr(k)**br

! set reaslistic limits on fallspeeds
! bug fix, 10/08/09
            dum=(rhosu/rho(k))**0.54
            umg=min(umg,20.*dum)
            ung=min(ung,20.*dum)
            umr=min(umr,9.1*dum)
            unr=min(unr,9.1*dum)

! pracg is mixing ratio of rain per sec collected by graupel/hail
            pracg(k) = cons41*(((1.2*umr-0.95*umg)**2+                   &
                  0.08*umg*umr)**0.5*rho(k)*                      &
                  n0rr(k)*n0g(k)/lamr(k)**3*                              &
                  (5./(lamr(k)**3*lamg(k))+                    &
                  2./(lamr(k)**2*lamg(k)**2)+				   &
				  0.5/(lamr(k)*lamg(k)**3)))

! assume 1 mm drops are shed, get number shed per sec

            dum = pracg(k)/5.2e-7

            npracg(k) = cons32*rho(k)*(1.7*(unr-ung)**2+            &
                0.3*unr*ung)**0.5*n0rr(k)*n0g(k)*              &
                (1./(lamr(k)**3*lamg(k))+                      &
                 1./(lamr(k)**2*lamg(k)**2)+                   &
                 1./(lamr(k)*lamg(k)**3))

! hm 7/15/13, remove limit so that the number of collected drops can smaller than 
! number of shed drops
!            npracg(k)=max(npracg(k)-dum,0.)
            npracg(k)=npracg(k)-dum

	    end if

!.......................................................................
! accretion of cloud liquid water by rain
! continuous collection equation with
! gravitational collection kernel, droplet fall speed neglected

         if (qr3d(k).ge.1.e-8 .and. qc3d(k).ge.1.e-8) then

! 12/13/06 hm add, replace with newer formula from
! khairoutdinov and kogan 2000, mwr

           dum=(qc3d(k)*qr3d(k))
           pra(k) = 67.*(dum)**1.15
           npra(k) = pra(k)/(qc3d(k)/nc3d(k))

         end if
!.......................................................................
! self-collection of rain drops
! from beheng(1994)
! from numerical simulation of the stochastic collection equation
! as descrined above for autoconversion

         if (qr3d(k).ge.1.e-8) then
! include breakup add 10/09/09
            dum1=300.e-6
            if (1./lamr(k).lt.dum1) then
            dum=1.
            else if (1./lamr(k).ge.dum1) then
            dum=2.-exp(2300.*(1./lamr(k)-dum1))
            end if
!            nragg(k) = -8.*nr3d(k)*qr3d(k)*rho(k)
            nragg(k) = -5.78*dum*nr3d(k)*qr3d(k)*rho(k)
         end if

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! calculate evap of rain (rutledge and hobbs 1983)

      if (qr3d(k).ge.qsmall) then
        epsr = 2.*pi*n0rr(k)*rho(k)*dv(k)*                           &
                   (f1r/(lamr(k)*lamr(k))+                       &
                    f2r*(arn(k)*rho(k)/mu(k))**0.5*                      &
                    sc(k)**(1./3.)*cons9/                   &
                (lamr(k)**cons34))
      else
      epsr = 0.
      end if

! no condensation onto rain, only evap allowed

           if (qv3d(k).lt.qvs(k)) then
              pre(k) = epsr*(qv3d(k)-qvs(k))/ab(k)
              pre(k) = min(pre(k),0.)
           else
              pre(k) = 0.
           end if

!.......................................................................
! melting of snow

! snow may persits above freezing, formula from rutledge and hobbs, 1984
! if water supersaturation, snow melts to form rain

          if (qni3d(k).ge.1.e-8) then

! fix 053011
! hm, modify for v3.2, add accelerated melting due to collision with rain
!             dum = -cpw/xlf(k)*t3d(k)*pracs(k)
             dum = -cpw/xlf(k)*(t3d(k)-273.15)*pracs(k)

! hm fix 1/20/15
!             psmlt(k)=2.*pi*n0s(k)*kap(k)*(273.15-t3d(k))/       &
!                    xlf(k)*rho(k)*(f1s/(lams(k)*lams(k))+        &
!                    f2s*(asn(k)*rho(k)/mu(k))**0.5*                      &
!                    sc(k)**(1./3.)*cons10/                   &
!                   (lams(k)**cons35))+dum
             psmlt(k)=2.*pi*n0s(k)*kap(k)*(273.15-t3d(k))/       &
                    xlf(k)*(f1s/(lams(k)*lams(k))+        &
                    f2s*(asn(k)*rho(k)/mu(k))**0.5*                      &
                    sc(k)**(1./3.)*cons10/                   &
                   (lams(k)**cons35))+dum

! in water subsaturation, snow melts and evaporates

      if (qvqvs(k).lt.1.) then
        epss = 2.*pi*n0s(k)*rho(k)*dv(k)*                            &
                   (f1s/(lams(k)*lams(k))+                       &
                    f2s*(asn(k)*rho(k)/mu(k))**0.5*                      &
                    sc(k)**(1./3.)*cons10/                   &
               (lams(k)**cons35))
! hm fix 8/4/08
        evpms(k) = (qv3d(k)-qvs(k))*epss/ab(k)    
        evpms(k) = max(evpms(k),psmlt(k))
        psmlt(k) = psmlt(k)-evpms(k)
      end if
      end if

!.......................................................................
! melting of graupel

! graupel may persits above freezing, formula from rutledge and hobbs, 1984
! if water supersaturation, graupel melts to form rain

          if (qg3d(k).ge.1.e-8) then

! fix 053011
! hm, modify for v3.2, add accelerated melting due to collision with rain
!             dum = -cpw/xlf(k)*t3d(k)*pracg(k)
             dum = -cpw/xlf(k)*(t3d(k)-273.15)*pracg(k)

! hm fix 1/20/15
!             pgmlt(k)=2.*pi*n0g(k)*kap(k)*(273.15-t3d(k))/ 		 &
!                    xlf(k)*rho(k)*(f1s/(lamg(k)*lamg(k))+                &
!                    f2s*(agn(k)*rho(k)/mu(k))**0.5*                      &
!                    sc(k)**(1./3.)*cons11/                   &
!                   (lamg(k)**cons36))+dum
             pgmlt(k)=2.*pi*n0g(k)*kap(k)*(273.15-t3d(k))/ 		 &
                    xlf(k)*(f1s/(lamg(k)*lamg(k))+                &
                    f2s*(agn(k)*rho(k)/mu(k))**0.5*                      &
                    sc(k)**(1./3.)*cons11/                   &
                   (lamg(k)**cons36))+dum

! in water subsaturation, graupel melts and evaporates

      if (qvqvs(k).lt.1.) then
        epsg = 2.*pi*n0g(k)*rho(k)*dv(k)*                                &
                   (f1s/(lamg(k)*lamg(k))+                               &
                    f2s*(agn(k)*rho(k)/mu(k))**0.5*                      &
                    sc(k)**(1./3.)*cons11/                   &
               (lamg(k)**cons36))
! hm fix 8/4/08
        evpmg(k) = (qv3d(k)-qvs(k))*epsg/ab(k)
        evpmg(k) = max(evpmg(k),pgmlt(k))
        pgmlt(k) = pgmlt(k)-evpmg(k)
      end if
      end if

! hm, v3.2
! reset pracg and pracs to zero, this is done because there is no
! transfer of mass from snow and graupel to rain directly from collection
! above freezing, it is only used for enhancement of melting and shedding

      pracg(k) = 0.
      pracs(k) = 0.

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

! for cloud ice, only processes operating at t > 273.15 is
! melting, which is already conserved during process
! calculation

! conservation of qc

      dum = (prc(k)+pra(k))*dt

      if (dum.gt.qc3d(k).and.qc3d(k).ge.qsmall) then

        ratio = qc3d(k)/dum

        prc(k) = prc(k)*ratio
        pra(k) = pra(k)*ratio

        end if

! conservation of snow

        dum = (-psmlt(k)-evpms(k)+pracs(k))*dt

        if (dum.gt.qni3d(k).and.qni3d(k).ge.qsmall) then

! no source terms for snow at t > freezing
        ratio = qni3d(k)/dum

        psmlt(k) = psmlt(k)*ratio
        evpms(k) = evpms(k)*ratio
        pracs(k) = pracs(k)*ratio

        end if

! conservation of graupel

        dum = (-pgmlt(k)-evpmg(k)+pracg(k))*dt

        if (dum.gt.qg3d(k).and.qg3d(k).ge.qsmall) then

! no source term for graupel above freezing
        ratio = qg3d(k)/dum

        pgmlt(k) = pgmlt(k)*ratio
        evpmg(k) = evpmg(k)*ratio
        pracg(k) = pracg(k)*ratio

        end if

! conservation of qr
! hm 12/13/06, added conservation of rain since pre is negative

        dum = (-pracs(k)-pracg(k)-pre(k)-pra(k)-prc(k)+psmlt(k)+pgmlt(k))*dt

        if (dum.gt.qr3d(k).and.qr3d(k).ge.qsmall) then

        ratio = (qr3d(k)/dt+pracs(k)+pracg(k)+pra(k)+prc(k)-psmlt(k)-pgmlt(k))/ &
                        (-pre(k))
        pre(k) = pre(k)*ratio
        
        end if

!....................................

      qv3dten(k) = qv3dten(k)+(-pre(k)-evpms(k)-evpmg(k))

      t3dten(k) = t3dten(k)+(pre(k)*xxlv(k)+(evpms(k)+evpmg(k))*xxls(k)+&
                    (psmlt(k)+pgmlt(k)-pracs(k)-pracg(k))*xlf(k))/cpm(k)

      qc3dten(k) = qc3dten(k)+(-pra(k)-prc(k))
      qr3dten(k) = qr3dten(k)+(pre(k)+pra(k)+prc(k)-psmlt(k)-pgmlt(k)+pracs(k)+pracg(k))
      qni3dten(k) = qni3dten(k)+(psmlt(k)+evpms(k)-pracs(k))
      qg3dten(k) = qg3dten(k)+(pgmlt(k)+evpmg(k)-pracg(k))
! fix 053011
!      ns3dten(k) = ns3dten(k)-npracs(k)
! hm, bug fix 5/12/08, npracg is subtracted from nr not ng
!      ng3dten(k) = ng3dten(k)
      nc3dten(k) = nc3dten(k)+ (-npra(k)-nprc(k))
      nr3dten(k) = nr3dten(k)+ (nprc1(k)+nragg(k)-npracg(k))

! hm add, wrf-chem, add tendencies for c2prec

	c2prec(k) = pra(k)+prc(k)
      if (pre(k).lt.0.) then
         dum = pre(k)*dt/qr3d(k)
           dum = max(-1.,dum)
         nsubr(k) = dum*nr3d(k)/dt
      end if

        if (evpms(k)+psmlt(k).lt.0.) then
         dum = (evpms(k)+psmlt(k))*dt/qni3d(k)
           dum = max(-1.,dum)
         nsmlts(k) = dum*ns3d(k)/dt
        end if
        if (psmlt(k).lt.0.) then
          dum = psmlt(k)*dt/qni3d(k)
          dum = max(-1.0,dum)
          nsmltr(k) = dum*ns3d(k)/dt
        end if
        if (evpmg(k)+pgmlt(k).lt.0.) then
         dum = (evpmg(k)+pgmlt(k))*dt/qg3d(k)
           dum = max(-1.,dum)
         ngmltg(k) = dum*ng3d(k)/dt
        end if
        if (pgmlt(k).lt.0.) then
          dum = pgmlt(k)*dt/qg3d(k)
          dum = max(-1.0,dum)
          ngmltr(k) = dum*ng3d(k)/dt
        end if

         ns3dten(k) = ns3dten(k)+(nsmlts(k))
         ng3dten(k) = ng3dten(k)+(ngmltg(k))
         nr3dten(k) = nr3dten(k)+(nsubr(k)-nsmltr(k)-ngmltr(k))

 300  continue

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now calculate saturation adjustment to condense extra vapor above
! water saturation

      dumt = t3d(k)+dt*t3dten(k)
      dumqv = qv3d(k)+dt*qv3dten(k)
! hm, add fix for low pressure, 5/12/10
      dum=min(0.99*pres(k),polysvp(dumt,0))
      dumqss = ep_2*dum/(pres(k)-dum)
      dumqc = qc3d(k)+dt*qc3dten(k)
      dumqc = max(dumqc,0.)

! saturation adjustment for liquid

      dums = dumqv-dumqss
      pcc(k) = dums/(1.+xxlv(k)**2*dumqss/(cpm(k)*rv*dumt**2))/dt
      if (pcc(k)*dt+dumqc.lt.0.) then
           pcc(k) = -dumqc/dt
      end if

      qv3dten(k) = qv3dten(k)-pcc(k)
      t3dten(k) = t3dten(k)+pcc(k)*xxlv(k)/cpm(k)
      qc3dten(k) = qc3dten(k)+pcc(k)

#if (wrf_chem == 1)
      if( has_wetscav ) then
         evapprod(k) = - pre(k) - evpms(k) - evpmg(k)
         rainprod(k) = pra(k) + prc(k) + tqimelt(k)
      end if
#endif

!.......................................................................
! activation of cloud droplets
! activation of droplet currently not calculated
! droplet concentration is specified !!!!!

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! sublimate, melt, or evaporate number concentration
! this formulation assumes 1:1 ratio between mass loss and
! loss of number concentration

!     if (pcc(k).lt.0.) then
!        dum = pcc(k)*dt/qc3d(k)
!           dum = max(-1.,dum)
!        nsubc(k) = dum*nc3d(k)/dt
!     end if

! update tendencies

!        nc3dten(k) = nc3dten(k)+nsubc(k)

!.....................................................................
!.....................................................................
         else  ! temperature < 273.15

!......................................................................
!hm add, allow for constant droplet number
! inum = 0, predict droplet number
! inum = 1, set constant droplet number

         if (iinum.eq.1) then
! convert ndcnst from cm-3 to kg-1
            nc3d(k)=ndcnst*1.e6/rho(k)
         end if

! calculate size distribution parameters
! make sure number concentrations aren't negative

      ni3d(k) = max(0.,ni3d(k))
      ns3d(k) = max(0.,ns3d(k))
      nc3d(k) = max(0.,nc3d(k))
      nr3d(k) = max(0.,nr3d(k))
      ng3d(k) = max(0.,ng3d(k))

!......................................................................
! cloud ice

      if (qi3d(k).ge.qsmall) then
         lami(k) = (cons12*                 &
              ni3d(k)/qi3d(k))**(1./di)
         n0i(k) = ni3d(k)*lami(k)

! check for slope

! adjust vars

      if (lami(k).lt.lammini) then

      lami(k) = lammini

      n0i(k) = lami(k)**4*qi3d(k)/cons12

      ni3d(k) = n0i(k)/lami(k)
      else if (lami(k).gt.lammaxi) then
      lami(k) = lammaxi
      n0i(k) = lami(k)**4*qi3d(k)/cons12

      ni3d(k) = n0i(k)/lami(k)
      end if
      end if

!......................................................................
! rain

      if (qr3d(k).ge.qsmall) then
      lamr(k) = (pi*rhow*nr3d(k)/qr3d(k))**(1./3.)
      n0rr(k) = nr3d(k)*lamr(k)

! check for slope

! adjust vars

      if (lamr(k).lt.lamminr) then

      lamr(k) = lamminr

      n0rr(k) = lamr(k)**4*qr3d(k)/(pi*rhow)

      nr3d(k) = n0rr(k)/lamr(k)
      else if (lamr(k).gt.lammaxr) then
      lamr(k) = lammaxr
      n0rr(k) = lamr(k)**4*qr3d(k)/(pi*rhow)

      nr3d(k) = n0rr(k)/lamr(k)
      end if
      end if

!......................................................................
! cloud droplets

! martin et al. (1994) formula for pgam

      if (qc3d(k).ge.qsmall) then

         dum = pres(k)/(287.15*t3d(k))
         pgam(k)=0.0005714*(nc3d(k)/1.e6*dum)+0.2714
         pgam(k)=1./(pgam(k)**2)-1.
         pgam(k)=max(pgam(k),2.)
         pgam(k)=min(pgam(k),10.)

! calculate lamc

      lamc(k) = (cons26*nc3d(k)*gamma(pgam(k)+4.)/   &
                 (qc3d(k)*gamma(pgam(k)+1.)))**(1./3.)

! lammin, 60 micron diameter
! lammax, 1 micron

      lammin = (pgam(k)+1.)/60.e-6
      lammax = (pgam(k)+1.)/1.e-6

      if (lamc(k).lt.lammin) then
      lamc(k) = lammin

      nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+              &
                log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26
      else if (lamc(k).gt.lammax) then
      lamc(k) = lammax
      nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+              &
                log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26

      end if

! to calculate droplet freezing

        cdist1(k) = nc3d(k)/gamma(pgam(k)+1.)

      end if

!......................................................................
! snow

      if (qni3d(k).ge.qsmall) then
      lams(k) = (cons1*ns3d(k)/qni3d(k))**(1./ds)
      n0s(k) = ns3d(k)*lams(k)

! check for slope

! adjust vars

      if (lams(k).lt.lammins) then
      lams(k) = lammins
      n0s(k) = lams(k)**4*qni3d(k)/cons1

      ns3d(k) = n0s(k)/lams(k)

      else if (lams(k).gt.lammaxs) then

      lams(k) = lammaxs
      n0s(k) = lams(k)**4*qni3d(k)/cons1

      ns3d(k) = n0s(k)/lams(k)
      end if
      end if

!......................................................................
! graupel

      if (qg3d(k).ge.qsmall) then
      lamg(k) = (cons2*ng3d(k)/qg3d(k))**(1./dg)
      n0g(k) = ng3d(k)*lamg(k)

! check for slope

! adjust vars

      if (lamg(k).lt.lamming) then
      lamg(k) = lamming
      n0g(k) = lamg(k)**4*qg3d(k)/cons2

      ng3d(k) = n0g(k)/lamg(k)

      else if (lamg(k).gt.lammaxg) then

      lamg(k) = lammaxg
      n0g(k) = lamg(k)**4*qg3d(k)/cons2

      ng3d(k) = n0g(k)/lamg(k)
      end if
      end if

!.....................................................................
! zero out process rates

            mnuccc(k) = 0.
            nnuccc(k) = 0.
            prc(k) = 0.
            nprc(k) = 0.
            nprc1(k) = 0.
            nsagg(k) = 0.
            psacws(k) = 0.
            npsacws(k) = 0.
            psacwi(k) = 0.
            npsacwi(k) = 0.
            pracs(k) = 0.
            npracs(k) = 0.
            nmults(k) = 0.
            qmults(k) = 0.
            nmultr(k) = 0.
            qmultr(k) = 0.
            nmultg(k) = 0.
            qmultg(k) = 0.
            nmultrg(k) = 0.
            qmultrg(k) = 0.
            mnuccr(k) = 0.
            nnuccr(k) = 0.
            pra(k) = 0.
            npra(k) = 0.
            nragg(k) = 0.
            prci(k) = 0.
            nprci(k) = 0.
            prai(k) = 0.
            nprai(k) = 0.
            nnuccd(k) = 0.
            mnuccd(k) = 0.
            pcc(k) = 0.
            pre(k) = 0.
            prd(k) = 0.
            prds(k) = 0.
            eprd(k) = 0.
            eprds(k) = 0.
            nsubc(k) = 0.
            nsubi(k) = 0.
            nsubs(k) = 0.
            nsubr(k) = 0.
            piacr(k) = 0.
            niacr(k) = 0.
            praci(k) = 0.
            piacrs(k) = 0.
            niacrs(k) = 0.
            pracis(k) = 0.
! hm: add graupel processes
            pracg(k) = 0.
            psacr(k) = 0.
	    psacwg(k) = 0.
	    pgsacw(k) = 0.
            pgracs(k) = 0.
	    prdg(k) = 0.
	    eprdg(k) = 0.
	    npracg(k) = 0.
	    npsacwg(k) = 0.
	    nscng(k) = 0.
 	    ngracs(k) = 0.
	    nsubg(k) = 0.

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! calculation of microphysical process rates
! accretion/autoconversion/freezing/melting/coag.
!.......................................................................
! freezing of cloud droplets
! only allowed below -4 c
        if (qc3d(k).ge.qsmall .and. t3d(k).lt.269.15) then

! number of contact nuclei (m^-3) from meyers et al., 1992
! factor of 1000 is to convert from l^-1 to m^-3

! meyers curve

           nacnt = exp(-2.80+0.262*(273.15-t3d(k)))*1000.

! cooper curve
!        nacnt =  5.*exp(0.304*(273.15-t3d(k)))

! flecther
!     nacnt = 0.01*exp(0.6*(273.15-t3d(k)))

! contact freezing

! mean free path

            dum = 7.37*t3d(k)/(288.*10.*pres(k))/100.

! effective diffusivity of contact nuclei
! based on brownian diffusion

            dap(k) = cons37*t3d(k)*(1.+dum/rin)/mu(k)
 
           mnuccc(k) = cons38*dap(k)*nacnt*exp(log(cdist1(k))+   &
                   log(gamma(pgam(k)+5.))-4.*log(lamc(k)))
           nnuccc(k) = 2.*pi*dap(k)*nacnt*cdist1(k)*           &
                    gamma(pgam(k)+2.)/                         &
                    lamc(k)

! immersion freezing (bigg 1953)

!           mnuccc(k) = mnuccc(k)+cons39*                   &
!                  exp(log(cdist1(k))+log(gamma(7.+pgam(k)))-6.*log(lamc(k)))*             &
!                   exp(aimm*(273.15-t3d(k)))

!           nnuccc(k) = nnuccc(k)+                                  &
!            cons40*exp(log(cdist1(k))+log(gamma(pgam(k)+4.))-3.*log(lamc(k)))              &
!                *exp(aimm*(273.15-t3d(k)))

! hm 7/15/13 fix for consistency w/ original formula
           mnuccc(k) = mnuccc(k)+cons39*                   &
                  exp(log(cdist1(k))+log(gamma(7.+pgam(k)))-6.*log(lamc(k)))*             &
                   (exp(aimm*(273.15-t3d(k)))-1.)

           nnuccc(k) = nnuccc(k)+                                  &
            cons40*exp(log(cdist1(k))+log(gamma(pgam(k)+4.))-3.*log(lamc(k)))              &
                *(exp(aimm*(273.15-t3d(k)))-1.)

! put in a catch here to prevent divergence between number conc. and
! mixing ratio, since strict conservation not checked for number conc

           nnuccc(k) = min(nnuccc(k),nc3d(k)/dt)

        end if

!.................................................................
!.......................................................................
! autoconversion of cloud liquid water to rain
! formula from beheng (1994)
! using numerical simulation of stochastic collection equation
! and initial cloud droplet size distribution specified
! as a gamma distribution

! use minimum value of 1.e-6 to prevent floating point error

         if (qc3d(k).ge.1.e-6) then

! hm add 12/13/06, replace with newer formula
! from khairoutdinov and kogan 2000, mwr

                prc(k)=1350.*qc3d(k)**2.47*  &
           (nc3d(k)/1.e6*rho(k))**(-1.79)

! note: nprc1 is change in nr,
! nprc is change in nc

        nprc1(k) = prc(k)/cons29
        nprc(k) = prc(k)/(qc3d(k)/nc3d(k))

! hm bug fix 3/20/12
                nprc(k) = min(nprc(k),nc3d(k)/dt)
                nprc1(k) = min(nprc1(k),nprc(k))

         end if

!.......................................................................
! self-collection of droplet not included in kk2000 scheme

! snow aggregation from passarelli, 1978, used by reisner, 1998
! this is hard-wired for bs = 0.4 for now

         if (qni3d(k).ge.1.e-8) then
             nsagg(k) = cons15*asn(k)*rho(k)**            &
            ((2.+bs)/3.)*qni3d(k)**((2.+bs)/3.)*                  &
            (ns3d(k)*rho(k))**((4.-bs)/3.)/                       &
            (rho(k))
         end if

!.......................................................................
! accretion of cloud droplets onto snow/graupel
! here use continuous collection equation with
! simple gravitational collection kernel ignoring

! snow

         if (qni3d(k).ge.1.e-8 .and. qc3d(k).ge.qsmall) then

           psacws(k) = cons13*asn(k)*qc3d(k)*rho(k)*               &
                  n0s(k)/                        &
                  lams(k)**(bs+3.)
           npsacws(k) = cons13*asn(k)*nc3d(k)*rho(k)*              &
                  n0s(k)/                        &
                  lams(k)**(bs+3.)

         end if

!............................................................................
! collection of cloud water by graupel

         if (qg3d(k).ge.1.e-8 .and. qc3d(k).ge.qsmall) then

           psacwg(k) = cons14*agn(k)*qc3d(k)*rho(k)*               &
                  n0g(k)/                        &
                  lamg(k)**(bg+3.)
           npsacwg(k) = cons14*agn(k)*nc3d(k)*rho(k)*              &
                  n0g(k)/                        &
                  lamg(k)**(bg+3.)
	    end if

!.......................................................................
! hm, add 12/13/06
! cloud ice collecting droplets, assume that cloud ice mean diam > 100 micron
! before riming can occur
! assume that rime collected on cloud ice does not lead
! to hallet-mossop splintering

         if (qi3d(k).ge.1.e-8 .and. qc3d(k).ge.qsmall) then

! put in size dependent collection efficiency based on stokes law
! from thompson et al. 2004, mwr

            if (1./lami(k).ge.100.e-6) then

           psacwi(k) = cons16*ain(k)*qc3d(k)*rho(k)*               &
                  n0i(k)/                        &
                  lami(k)**(bi+3.)
           npsacwi(k) = cons16*ain(k)*nc3d(k)*rho(k)*              &
                  n0i(k)/                        &
                  lami(k)**(bi+3.)
           end if
         end if

!.......................................................................
! accretion of rain water by snow
! formula from ikawa and saito, 1991, used by reisner et al, 1998

         if (qr3d(k).ge.1.e-8.and.qni3d(k).ge.1.e-8) then

            ums = asn(k)*cons3/(lams(k)**bs)
            umr = arn(k)*cons4/(lamr(k)**br)
            uns = asn(k)*cons5/lams(k)**bs
            unr = arn(k)*cons6/lamr(k)**br

! set reaslistic limits on fallspeeds

! bug fix, 10/08/09
            dum=(rhosu/rho(k))**0.54
            ums=min(ums,1.2*dum)
            uns=min(uns,1.2*dum)
            umr=min(umr,9.1*dum)
            unr=min(unr,9.1*dum)

            pracs(k) = cons41*(((1.2*umr-0.95*ums)**2+                   &
                  0.08*ums*umr)**0.5*rho(k)*                      &
                  n0rr(k)*n0s(k)/lamr(k)**3*                              &
                  (5./(lamr(k)**3*lams(k))+                    &
                  2./(lamr(k)**2*lams(k)**2)+                  &				 
                  0.5/(lamr(k)*lams(k)**3)))

            npracs(k) = cons32*rho(k)*(1.7*(unr-uns)**2+            &
                0.3*unr*uns)**0.5*n0rr(k)*n0s(k)*              &
                (1./(lamr(k)**3*lams(k))+                      &
                 1./(lamr(k)**2*lams(k)**2)+                   &
                 1./(lamr(k)*lams(k)**3))

! make sure pracs doesn't exceed total rain mixing ratio
! as this may otherwise result in too much transfer of water during
! rime-splintering

            pracs(k) = min(pracs(k),qr3d(k)/dt)

! collection of snow by rain - needed for graupel conversion calculations
! only calculate if snow and rain mixing ratios exceed 0.1 g/kg

! hm modify for wrfv3.1
!            if (ihail.eq.0) then
            if (qni3d(k).ge.0.1e-3.and.qr3d(k).ge.0.1e-3) then
            psacr(k) = cons31*(((1.2*umr-0.95*ums)**2+              &
                  0.08*ums*umr)**0.5*rho(k)*                     &
                 n0rr(k)*n0s(k)/lams(k)**3*                               &
                  (5./(lams(k)**3*lamr(k))+                    &
                  2./(lams(k)**2*lamr(k)**2)+                  &
                  0.5/(lams(k)*lamr(k)**3)))            
            end if
!            end if

         end if

!.......................................................................

! collection of rainwater by graupel, from ikawa and saito 1990, 
! used by reisner et al 1998
         if (qr3d(k).ge.1.e-8.and.qg3d(k).ge.1.e-8) then

            umg = agn(k)*cons7/(lamg(k)**bg)
            umr = arn(k)*cons4/(lamr(k)**br)
            ung = agn(k)*cons8/lamg(k)**bg
            unr = arn(k)*cons6/lamr(k)**br

! set reaslistic limits on fallspeeds
! bug fix, 10/08/09
            dum=(rhosu/rho(k))**0.54
            umg=min(umg,20.*dum)
            ung=min(ung,20.*dum)
            umr=min(umr,9.1*dum)
            unr=min(unr,9.1*dum)

            pracg(k) = cons41*(((1.2*umr-0.95*umg)**2+                   &
                  0.08*umg*umr)**0.5*rho(k)*                      &
                  n0rr(k)*n0g(k)/lamr(k)**3*                              &
                  (5./(lamr(k)**3*lamg(k))+                    &
                  2./(lamr(k)**2*lamg(k)**2)+				   &
				  0.5/(lamr(k)*lamg(k)**3)))

            npracg(k) = cons32*rho(k)*(1.7*(unr-ung)**2+            &
                0.3*unr*ung)**0.5*n0rr(k)*n0g(k)*              &
                (1./(lamr(k)**3*lamg(k))+                      &
                 1./(lamr(k)**2*lamg(k)**2)+                   &
                 1./(lamr(k)*lamg(k)**3))

! make sure pracg doesn't exceed total rain mixing ratio
! as this may otherwise result in too much transfer of water during
! rime-splintering

            pracg(k) = min(pracg(k),qr3d(k)/dt)

	    end if

!.......................................................................
! rime-splintering - snow
! hallet-mossop (1974)
! number of splinters formed is based on mass of rimed water

! dum1 = mass of individual splinters

! hm add threshold snow and droplet mixing ratio for rime-splintering
! to limit rime-splintering in stratiform clouds
! these thresholds correspond with graupel thresholds in rh 1984

!v1.4
         if (qni3d(k).ge.0.1e-3) then
         if (qc3d(k).ge.0.5e-3.or.qr3d(k).ge.0.1e-3) then
         if (psacws(k).gt.0..or.pracs(k).gt.0.) then
            if (t3d(k).lt.270.16 .and. t3d(k).gt.265.16) then

               if (t3d(k).gt.270.16) then
                  fmult = 0.
               else if (t3d(k).le.270.16.and.t3d(k).gt.268.16)  then
                  fmult = (270.16-t3d(k))/2.
               else if (t3d(k).ge.265.16.and.t3d(k).le.268.16)   then
                  fmult = (t3d(k)-265.16)/3.
               else if (t3d(k).lt.265.16) then
                  fmult = 0.
               end if

! 1000 is to convert from kg to g

! splintering from droplets accreted onto snow

               if (psacws(k).gt.0.) then
                  nmults(k) = 35.e4*psacws(k)*fmult*1000.
                  qmults(k) = nmults(k)*mmult

! constrain so that transfer of mass from snow to ice cannot be more mass
! than was rimed onto snow

                  qmults(k) = min(qmults(k),psacws(k))
                  psacws(k) = psacws(k)-qmults(k)

               end if

! riming and splintering from accreted raindrops

               if (pracs(k).gt.0.) then
                   nmultr(k) = 35.e4*pracs(k)*fmult*1000.
                   qmultr(k) = nmultr(k)*mmult

! constrain so that transfer of mass from snow to ice cannot be more mass
! than was rimed onto snow

                   qmultr(k) = min(qmultr(k),pracs(k))

                   pracs(k) = pracs(k)-qmultr(k)

               end if

            end if
         end if
         end if
         end if

!.......................................................................
! rime-splintering - graupel 
! hallet-mossop (1974)
! number of splinters formed is based on mass of rimed water

! dum1 = mass of individual splinters

! hm add threshold snow mixing ratio for rime-splintering
! to limit rime-splintering in stratiform clouds

!         if (ihail.eq.0) then
! v1.4
         if (qg3d(k).ge.0.1e-3) then
         if (qc3d(k).ge.0.5e-3.or.qr3d(k).ge.0.1e-3) then
         if (psacwg(k).gt.0..or.pracg(k).gt.0.) then
            if (t3d(k).lt.270.16 .and. t3d(k).gt.265.16) then

               if (t3d(k).gt.270.16) then
                  fmult = 0.
               else if (t3d(k).le.270.16.and.t3d(k).gt.268.16)  then
                  fmult = (270.16-t3d(k))/2.
               else if (t3d(k).ge.265.16.and.t3d(k).le.268.16)   then
                  fmult = (t3d(k)-265.16)/3.
               else if (t3d(k).lt.265.16) then
                  fmult = 0.
               end if

! 1000 is to convert from kg to g

! splintering from droplets accreted onto graupel

               if (psacwg(k).gt.0.) then
                  nmultg(k) = 35.e4*psacwg(k)*fmult*1000.
                  qmultg(k) = nmultg(k)*mmult

! constrain so that transfer of mass from graupel to ice cannot be more mass
! than was rimed onto graupel

                  qmultg(k) = min(qmultg(k),psacwg(k))
                  psacwg(k) = psacwg(k)-qmultg(k)

               end if

! riming and splintering from accreted raindrops

               if (pracg(k).gt.0.) then
                   nmultrg(k) = 35.e4*pracg(k)*fmult*1000.
                   qmultrg(k) = nmultrg(k)*mmult

! constrain so that transfer of mass from graupel to ice cannot be more mass
! than was rimed onto graupel

                   qmultrg(k) = min(qmultrg(k),pracg(k))
                   pracg(k) = pracg(k)-qmultrg(k)

               end if
               end if
               end if
            end if
            end if
!         end if

!........................................................................
! conversion of rimed cloud water onto snow to graupel/hail

!           if (ihail.eq.0) then
	   if (psacws(k).gt.0.) then
! only allow conversion if qni > 0.1 and qc > 0.5 g/kg following rutledge and hobbs (1984)
              if (qni3d(k).ge.0.1e-3.and.qc3d(k).ge.0.5e-3) then

! portion of riming converted to graupel (reisner et al. 1998, originally is1991)
	     pgsacw(k) = min(psacws(k),cons17*dt*n0s(k)*qc3d(k)*qc3d(k)* &
                          asn(k)*asn(k)/ &
                           (rho(k)*lams(k)**(2.*bs+2.))) 

! mix rat converted into graupel as embryo (reisner et al. 1998, orig m1990)
	     dum = max(rhosn/(rhog-rhosn)*pgsacw(k),0.) 

! number concentraiton of embryo graupel from riming of snow
	     nscng(k) = dum/mg0*rho(k)
! limit max number converted to snow number
             nscng(k) = min(nscng(k),ns3d(k)/dt)

! portion of riming left for snow
             psacws(k) = psacws(k) - pgsacw(k)
             end if
	   end if

! conversion of rimed rainwater onto snow converted to graupel

	   if (pracs(k).gt.0.) then
! only allow conversion if qni > 0.1 and qr > 0.1 g/kg following rutledge and hobbs (1984)
              if (qni3d(k).ge.0.1e-3.and.qr3d(k).ge.0.1e-3) then
! portion of collected rainwater converted to graupel (reisner et al. 1998)
	      dum = cons18*(4./lams(k))**3*(4./lams(k))**3 &    
                   /(cons18*(4./lams(k))**3*(4./lams(k))**3+ &  
                   cons19*(4./lamr(k))**3*(4./lamr(k))**3)
              dum=min(dum,1.)
              dum=max(dum,0.)
	      pgracs(k) = (1.-dum)*pracs(k)
            ngracs(k) = (1.-dum)*npracs(k)
! limit max number converted to min of either rain or snow number concentration
            ngracs(k) = min(ngracs(k),nr3d(k)/dt)
            ngracs(k) = min(ngracs(k),ns3d(k)/dt)

! amount left for snow production
            pracs(k) = pracs(k) - pgracs(k)
            npracs(k) = npracs(k) - ngracs(k)
! conversion to graupel due to collection of snow by rain
            psacr(k)=psacr(k)*(1.-dum)
            end if
	   end if
!           end if

!.......................................................................
! freezing of rain drops
! freezing allowed below -4 c

         if (t3d(k).lt.269.15.and.qr3d(k).ge.qsmall) then

! immersion freezing (bigg 1953)
!            mnuccr(k) = cons20*nr3d(k)*exp(aimm*(273.15-t3d(k)))/lamr(k)**3 &
!                 /lamr(k)**3

!            nnuccr(k) = pi*nr3d(k)*bimm*exp(aimm*(273.15-t3d(k)))/lamr(k)**3

! hm fix 7/15/13 for consistency w/ original formula
            mnuccr(k) = cons20*nr3d(k)*(exp(aimm*(273.15-t3d(k)))-1.)/lamr(k)**3 &
                 /lamr(k)**3

            nnuccr(k) = pi*nr3d(k)*bimm*(exp(aimm*(273.15-t3d(k)))-1.)/lamr(k)**3

! prevent divergence between mixing ratio and number conc
            nnuccr(k) = min(nnuccr(k),nr3d(k)/dt)

         end if

!.......................................................................
! accretion of cloud liquid water by rain
! continuous collection equation with
! gravitational collection kernel, droplet fall speed neglected

         if (qr3d(k).ge.1.e-8 .and. qc3d(k).ge.1.e-8) then

! 12/13/06 hm add, replace with newer formula from
! khairoutdinov and kogan 2000, mwr

           dum=(qc3d(k)*qr3d(k))
           pra(k) = 67.*(dum)**1.15
           npra(k) = pra(k)/(qc3d(k)/nc3d(k))

         end if
!.......................................................................
! self-collection of rain drops
! from beheng(1994)
! from numerical simulation of the stochastic collection equation
! as descrined above for autoconversion

         if (qr3d(k).ge.1.e-8) then
! include breakup add 10/09/09
            dum1=300.e-6
            if (1./lamr(k).lt.dum1) then
            dum=1.
            else if (1./lamr(k).ge.dum1) then
            dum=2.-exp(2300.*(1./lamr(k)-dum1))
            end if
!            nragg(k) = -8.*nr3d(k)*qr3d(k)*rho(k)
            nragg(k) = -5.78*dum*nr3d(k)*qr3d(k)*rho(k)
         end if

!.......................................................................
! autoconversion of cloud ice to snow
! following harrington et al. (1995) with modification
! here it is assumed that autoconversion can only occur when the
! ice is growing, i.e. in conditions of ice supersaturation

         if (qi3d(k).ge.1.e-8 .and.qvqvsi(k).ge.1.) then

!           coffi = 2./lami(k)
!           if (coffi.ge.dcs) then
              nprci(k) = cons21*(qv3d(k)-qvi(k))*rho(k)                         &
                *n0i(k)*exp(-lami(k)*dcs)*dv(k)/abi(k)
              prci(k) = cons22*nprci(k)
              nprci(k) = min(nprci(k),ni3d(k)/dt)

!           end if
         end if

!.......................................................................
! accretion of cloud ice by snow
! for this calculation, it is assumed that the vs >> vi
! and ds >> di for continuous collection

         if (qni3d(k).ge.1.e-8 .and. qi3d(k).ge.qsmall) then
            prai(k) = cons23*asn(k)*qi3d(k)*rho(k)*n0s(k)/     &
                     lams(k)**(bs+3.)
            nprai(k) = cons23*asn(k)*ni3d(k)*                                       &
                  rho(k)*n0s(k)/                                 &
                  lams(k)**(bs+3.)
            nprai(k)=min(nprai(k),ni3d(k)/dt)
         end if

!.......................................................................
! hm, add 12/13/06, collision of rain and ice to produce snow or graupel
! follows reisner et al. 1998
! assumed fallspeed and size of ice crystal << than for rain

         if (qr3d(k).ge.1.e-8.and.qi3d(k).ge.1.e-8.and.t3d(k).le.273.15) then

! allow graupel formation from rain-ice collisions only if rain mixing ratio > 0.1 g/kg,
! otherwise add to snow

            if (qr3d(k).ge.0.1e-3) then
            niacr(k)=cons24*ni3d(k)*n0rr(k)*arn(k) &
                /lamr(k)**(br+3.)*rho(k)
            piacr(k)=cons25*ni3d(k)*n0rr(k)*arn(k) &
                /lamr(k)**(br+3.)/lamr(k)**3*rho(k)
            praci(k)=cons24*qi3d(k)*n0rr(k)*arn(k)/ &
                lamr(k)**(br+3.)*rho(k)
            niacr(k)=min(niacr(k),nr3d(k)/dt)
            niacr(k)=min(niacr(k),ni3d(k)/dt)
            else 
            niacrs(k)=cons24*ni3d(k)*n0rr(k)*arn(k) &
                /lamr(k)**(br+3.)*rho(k)
            piacrs(k)=cons25*ni3d(k)*n0rr(k)*arn(k) &
                /lamr(k)**(br+3.)/lamr(k)**3*rho(k)
            pracis(k)=cons24*qi3d(k)*n0rr(k)*arn(k)/ &
                lamr(k)**(br+3.)*rho(k)
            niacrs(k)=min(niacrs(k),nr3d(k)/dt)
            niacrs(k)=min(niacrs(k),ni3d(k)/dt)
            end if
         end if

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! nucleation of cloud ice from homogeneous and heterogeneous freezing on aerosol

         if (inuc.eq.0) then

! add threshold according to greg thomspon

         if ((qvqvs(k).ge.0.999.and.t3d(k).le.265.15).or. &
              qvqvsi(k).ge.1.08) then

! hm, modify dec. 5, 2006, replace with cooper curve
      kc2 = 0.005*exp(0.304*(273.15-t3d(k)))*1000. ! convert from l-1 to m-3
! limit to 500 l-1
      kc2 = min(kc2,500.e3)
      kc2=max(kc2/rho(k),0.)  ! convert to kg-1

          if (kc2.gt.ni3d(k)+ns3d(k)+ng3d(k)) then
             nnuccd(k) = (kc2-ni3d(k)-ns3d(k)-ng3d(k))/dt
             mnuccd(k) = nnuccd(k)*mi0
          end if

          end if

          else if (inuc.eq.1) then

          if (t3d(k).lt.273.15.and.qvqvsi(k).gt.1.) then

             kc2 = 0.16*1000./rho(k)  ! convert from l-1 to kg-1
          if (kc2.gt.ni3d(k)+ns3d(k)+ng3d(k)) then
             nnuccd(k) = (kc2-ni3d(k)-ns3d(k)-ng3d(k))/dt
             mnuccd(k) = nnuccd(k)*mi0
          end if
          end if

         end if

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

 101      continue

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! calculate evap/sub/dep terms for qi,qni,qr

! no ventilation for cloud ice

        if (qi3d(k).ge.qsmall) then

         epsi = 2.*pi*n0i(k)*rho(k)*dv(k)/(lami(k)*lami(k))

      else
         epsi = 0.
      end if

      if (qni3d(k).ge.qsmall) then
        epss = 2.*pi*n0s(k)*rho(k)*dv(k)*                            &
                   (f1s/(lams(k)*lams(k))+                       &
                    f2s*(asn(k)*rho(k)/mu(k))**0.5*                      &
                    sc(k)**(1./3.)*cons10/                   &
               (lams(k)**cons35))
      else
      epss = 0.
      end if

      if (qg3d(k).ge.qsmall) then
        epsg = 2.*pi*n0g(k)*rho(k)*dv(k)*                                &
                   (f1s/(lamg(k)*lamg(k))+                               &
                    f2s*(agn(k)*rho(k)/mu(k))**0.5*                      &
                    sc(k)**(1./3.)*cons11/                   &
               (lamg(k)**cons36))


      else
      epsg = 0.
      end if

      if (qr3d(k).ge.qsmall) then
        epsr = 2.*pi*n0rr(k)*rho(k)*dv(k)*                           &
                   (f1r/(lamr(k)*lamr(k))+                       &
                    f2r*(arn(k)*rho(k)/mu(k))**0.5*                      &
                    sc(k)**(1./3.)*cons9/                   &
                (lamr(k)**cons34))
      else
      epsr = 0.
      end if

! only include region of ice size dist < dcs
! dum is fraction of d*n(d) < dcs

! logic below follows that of harrington et al. 1995 (jas)
              if (qi3d(k).ge.qsmall) then              
              dum=(1.-exp(-lami(k)*dcs)*(1.+lami(k)*dcs))
              prd(k) = epsi*(qv3d(k)-qvi(k))/abi(k)*dum
              else
              dum=0.
              end if
! add deposition in tail of ice size dist to snow if snow is present
              if (qni3d(k).ge.qsmall) then
              prds(k) = epss*(qv3d(k)-qvi(k))/abi(k)+ &
                epsi*(qv3d(k)-qvi(k))/abi(k)*(1.-dum)
! otherwise add to cloud ice
              else
              prd(k) = prd(k)+epsi*(qv3d(k)-qvi(k))/abi(k)*(1.-dum)
              end if
! vapor dpeosition on graupel
              prdg(k) = epsg*(qv3d(k)-qvi(k))/abi(k)

! no condensation onto rain, only evap

           if (qv3d(k).lt.qvs(k)) then
              pre(k) = epsr*(qv3d(k)-qvs(k))/ab(k)
              pre(k) = min(pre(k),0.)
           else
              pre(k) = 0.
           end if

! make sure not pushed into ice supersat/subsat
! formula from reisner 2 scheme

           dum = (qv3d(k)-qvi(k))/dt

           fudgef = 0.9999
           sum_dep = prd(k)+prds(k)+mnuccd(k)+prdg(k)

           if( (dum.gt.0. .and. sum_dep.gt.dum*fudgef) .or.                      &
               (dum.lt.0. .and. sum_dep.lt.dum*fudgef) ) then
               mnuccd(k) = fudgef*mnuccd(k)*dum/sum_dep
               prd(k) = fudgef*prd(k)*dum/sum_dep
               prds(k) = fudgef*prds(k)*dum/sum_dep
	       prdg(k) = fudgef*prdg(k)*dum/sum_dep
           endif

! if cloud ice/snow/graupel vap deposition is neg, then assign to sublimation processes

           if (prd(k).lt.0.) then
              eprd(k)=prd(k)
              prd(k)=0.
           end if
           if (prds(k).lt.0.) then
              eprds(k)=prds(k)
              prds(k)=0.
           end if
           if (prdg(k).lt.0.) then
              eprdg(k)=prdg(k)
              prdg(k)=0.
           end if

!.......................................................................
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

! conservation of water
! this is adopted loosely from mm5 resiner code. however, here we
! only adjust processes that are negative, rather than all processes.

! if mixing ratios less than qsmall, then no depletion of water
! through microphysical processes, skip conservation

! note: conservation check not applied to number concentration species. additional catch
! below will prevent negative number concentration
! for each microphysical process which provides a source for number, there is a check
! to make sure that can't exceed total number of depleted species with the time
! step

!****sensitivity - no ice

      if (iliq.eq.1) then
      mnuccc(k)=0.
      nnuccc(k)=0.
      mnuccr(k)=0.
      nnuccr(k)=0.
      mnuccd(k)=0.
      nnuccd(k)=0.
      end if

! ****sensitivity - no graupel
      if (igraup.eq.1) then
            pracg(k) = 0.
            psacr(k) = 0.
	    psacwg(k) = 0.
	    prdg(k) = 0.
	    eprdg(k) = 0.
            evpmg(k) = 0.
            pgmlt(k) = 0.
	    npracg(k) = 0.
	    npsacwg(k) = 0.
	    nscng(k) = 0.
 	    ngracs(k) = 0.
	    nsubg(k) = 0.
	    ngmltg(k) = 0.
            ngmltr(k) = 0.
! fix 053011
            piacrs(k)=piacrs(k)+piacr(k)
            piacr(k) = 0.
! fix 070713
	    pracis(k)=pracis(k)+praci(k)
	    praci(k) = 0.
	    psacws(k)=psacws(k)+pgsacw(k)
	    pgsacw(k) = 0.
	    pracs(k)=pracs(k)+pgracs(k)
	    pgracs(k) = 0.
       end if

! conservation of qc

      dum = (prc(k)+pra(k)+mnuccc(k)+psacws(k)+psacwi(k)+qmults(k)+psacwg(k)+pgsacw(k)+qmultg(k))*dt

      if (dum.gt.qc3d(k).and.qc3d(k).ge.qsmall) then
        ratio = qc3d(k)/dum

        prc(k) = prc(k)*ratio
        pra(k) = pra(k)*ratio
        mnuccc(k) = mnuccc(k)*ratio
        psacws(k) = psacws(k)*ratio
        psacwi(k) = psacwi(k)*ratio
        qmults(k) = qmults(k)*ratio
        qmultg(k) = qmultg(k)*ratio
        psacwg(k) = psacwg(k)*ratio
	pgsacw(k) = pgsacw(k)*ratio
        end if
 
! conservation of qi

      dum = (-prd(k)-mnuccc(k)+prci(k)+prai(k)-qmults(k)-qmultg(k)-qmultr(k)-qmultrg(k) &
                -mnuccd(k)+praci(k)+pracis(k)-eprd(k)-psacwi(k))*dt

      if (dum.gt.qi3d(k).and.qi3d(k).ge.qsmall) then

        ratio = (qi3d(k)/dt+prd(k)+mnuccc(k)+qmults(k)+qmultg(k)+qmultr(k)+qmultrg(k)+ &
                     mnuccd(k)+psacwi(k))/ &
                      (prci(k)+prai(k)+praci(k)+pracis(k)-eprd(k))

        prci(k) = prci(k)*ratio
        prai(k) = prai(k)*ratio
        praci(k) = praci(k)*ratio
        pracis(k) = pracis(k)*ratio
        eprd(k) = eprd(k)*ratio

        end if

! conservation of qr

      dum=((pracs(k)-pre(k))+(qmultr(k)+qmultrg(k)-prc(k))+(mnuccr(k)-pra(k))+ &
             piacr(k)+piacrs(k)+pgracs(k)+pracg(k))*dt

      if (dum.gt.qr3d(k).and.qr3d(k).ge.qsmall) then

        ratio = (qr3d(k)/dt+prc(k)+pra(k))/ &
             (-pre(k)+qmultr(k)+qmultrg(k)+pracs(k)+mnuccr(k)+piacr(k)+piacrs(k)+pgracs(k)+pracg(k))

        pre(k) = pre(k)*ratio
        pracs(k) = pracs(k)*ratio
        qmultr(k) = qmultr(k)*ratio
        qmultrg(k) = qmultrg(k)*ratio
        mnuccr(k) = mnuccr(k)*ratio
        piacr(k) = piacr(k)*ratio
        piacrs(k) = piacrs(k)*ratio
        pgracs(k) = pgracs(k)*ratio
        pracg(k) = pracg(k)*ratio

        end if

! conservation of qni
! conservation for graupel scheme

        if (igraup.eq.0) then

      dum = (-prds(k)-psacws(k)-prai(k)-prci(k)-pracs(k)-eprds(k)+psacr(k)-piacrs(k)-pracis(k))*dt

      if (dum.gt.qni3d(k).and.qni3d(k).ge.qsmall) then

        ratio = (qni3d(k)/dt+prds(k)+psacws(k)+prai(k)+prci(k)+pracs(k)+piacrs(k)+pracis(k))/(-eprds(k)+psacr(k))

       eprds(k) = eprds(k)*ratio
       psacr(k) = psacr(k)*ratio

       end if

! for no graupel, need to include freezing of rain for snow
       else if (igraup.eq.1) then

      dum = (-prds(k)-psacws(k)-prai(k)-prci(k)-pracs(k)-eprds(k)+psacr(k)-piacrs(k)-pracis(k)-mnuccr(k))*dt

      if (dum.gt.qni3d(k).and.qni3d(k).ge.qsmall) then

       ratio = (qni3d(k)/dt+prds(k)+psacws(k)+prai(k)+prci(k)+pracs(k)+piacrs(k)+pracis(k)+mnuccr(k))/(-eprds(k)+psacr(k))

       eprds(k) = eprds(k)*ratio
       psacr(k) = psacr(k)*ratio

       end if

       end if

! conservation of qg

      dum = (-psacwg(k)-pracg(k)-pgsacw(k)-pgracs(k)-prdg(k)-mnuccr(k)-eprdg(k)-piacr(k)-praci(k)-psacr(k))*dt

      if (dum.gt.qg3d(k).and.qg3d(k).ge.qsmall) then

        ratio = (qg3d(k)/dt+psacwg(k)+pracg(k)+pgsacw(k)+pgracs(k)+prdg(k)+mnuccr(k)+psacr(k)+&
                  piacr(k)+praci(k))/(-eprdg(k))

       eprdg(k) = eprdg(k)*ratio

      end if

! tendencies

      qv3dten(k) = qv3dten(k)+(-pre(k)-prd(k)-prds(k)-mnuccd(k)-eprd(k)-eprds(k)-prdg(k)-eprdg(k))

! bug fix hm, 3/1/11, include piacr and piacrs
      t3dten(k) = t3dten(k)+(pre(k)                                 &
               *xxlv(k)+(prd(k)+prds(k)+                            &
                mnuccd(k)+eprd(k)+eprds(k)+prdg(k)+eprdg(k))*xxls(k)+         &
               (psacws(k)+psacwi(k)+mnuccc(k)+mnuccr(k)+                      &
                qmults(k)+qmultg(k)+qmultr(k)+qmultrg(k)+pracs(k) &
                +psacwg(k)+pracg(k)+pgsacw(k)+pgracs(k)+piacr(k)+piacrs(k))*xlf(k))/cpm(k)

      qc3dten(k) = qc3dten(k)+                                      &
                 (-pra(k)-prc(k)-mnuccc(k)+pcc(k)-                  &
                  psacws(k)-psacwi(k)-qmults(k)-qmultg(k)-psacwg(k)-pgsacw(k))
      qi3dten(k) = qi3dten(k)+                                      &
         (prd(k)+eprd(k)+psacwi(k)+mnuccc(k)-prci(k)-                                 &
                  prai(k)+qmults(k)+qmultg(k)+qmultr(k)+qmultrg(k)+mnuccd(k)-praci(k)-pracis(k))
      qr3dten(k) = qr3dten(k)+                                      &
                 (pre(k)+pra(k)+prc(k)-pracs(k)-mnuccr(k)-qmultr(k)-qmultrg(k) &
             -piacr(k)-piacrs(k)-pracg(k)-pgracs(k))

      if (igraup.eq.0) then

      qni3dten(k) = qni3dten(k)+                                    &
           (prai(k)+psacws(k)+prds(k)+pracs(k)+prci(k)+eprds(k)-psacr(k)+piacrs(k)+pracis(k))
      ns3dten(k) = ns3dten(k)+(nsagg(k)+nprci(k)-nscng(k)-ngracs(k)+niacrs(k))
      qg3dten(k) = qg3dten(k)+(pracg(k)+psacwg(k)+pgsacw(k)+pgracs(k)+ &
                    prdg(k)+eprdg(k)+mnuccr(k)+piacr(k)+praci(k)+psacr(k))
      ng3dten(k) = ng3dten(k)+(nscng(k)+ngracs(k)+nnuccr(k)+niacr(k))

! for no graupel, need to include freezing of rain for snow
      else if (igraup.eq.1) then

      qni3dten(k) = qni3dten(k)+                                    &
           (prai(k)+psacws(k)+prds(k)+pracs(k)+prci(k)+eprds(k)-psacr(k)+piacrs(k)+pracis(k)+mnuccr(k))
      ns3dten(k) = ns3dten(k)+(nsagg(k)+nprci(k)-nscng(k)-ngracs(k)+niacrs(k)+nnuccr(k))

      end if

      nc3dten(k) = nc3dten(k)+(-nnuccc(k)-npsacws(k)                &
            -npra(k)-nprc(k)-npsacwi(k)-npsacwg(k))

      ni3dten(k) = ni3dten(k)+                                      &
       (nnuccc(k)-nprci(k)-nprai(k)+nmults(k)+nmultg(k)+nmultr(k)+nmultrg(k)+ &
               nnuccd(k)-niacr(k)-niacrs(k))

      nr3dten(k) = nr3dten(k)+(nprc1(k)-npracs(k)-nnuccr(k)      &
                   +nragg(k)-niacr(k)-niacrs(k)-npracg(k)-ngracs(k))

! hm add, wrf-chem, add tendencies for c2prec

	c2prec(k) = pra(k)+prc(k)+psacws(k)+qmults(k)+qmultg(k)+psacwg(k)+ &
       pgsacw(k)+mnuccc(k)+psacwi(k)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now calculate saturation adjustment to condense extra vapor above
! water saturation

      dumt = t3d(k)+dt*t3dten(k)
      dumqv = qv3d(k)+dt*qv3dten(k)
! hm, add fix for low pressure, 5/12/10
      dum=min(0.99*pres(k),polysvp(dumt,0))
      dumqss = ep_2*dum/(pres(k)-dum)
      dumqc = qc3d(k)+dt*qc3dten(k)
      dumqc = max(dumqc,0.)

! saturation adjustment for liquid

      dums = dumqv-dumqss
      pcc(k) = dums/(1.+xxlv(k)**2*dumqss/(cpm(k)*rv*dumt**2))/dt
      if (pcc(k)*dt+dumqc.lt.0.) then
           pcc(k) = -dumqc/dt
      end if

      qv3dten(k) = qv3dten(k)-pcc(k)
      t3dten(k) = t3dten(k)+pcc(k)*xxlv(k)/cpm(k)
      qc3dten(k) = qc3dten(k)+pcc(k)

!.......................................................................
! activation of cloud droplets
! activation of droplet currently not calculated
! droplet concentration is specified !!!!!

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! sublimate, melt, or evaporate number concentration
! this formulation assumes 1:1 ratio between mass loss and
! loss of number concentration

!     if (pcc(k).lt.0.) then
!        dum = pcc(k)*dt/qc3d(k)
!           dum = max(-1.,dum)
!        nsubc(k) = dum*nc3d(k)/dt
!     end if

      if (eprd(k).lt.0.) then
         dum = eprd(k)*dt/qi3d(k)
            dum = max(-1.,dum)
         nsubi(k) = dum*ni3d(k)/dt
      end if
      if (eprds(k).lt.0.) then
         dum = eprds(k)*dt/qni3d(k)
           dum = max(-1.,dum)
         nsubs(k) = dum*ns3d(k)/dt
      end if
      if (pre(k).lt.0.) then
         dum = pre(k)*dt/qr3d(k)
           dum = max(-1.,dum)
         nsubr(k) = dum*nr3d(k)/dt
      end if
      if (eprdg(k).lt.0.) then
         dum = eprdg(k)*dt/qg3d(k)
           dum = max(-1.,dum)
         nsubg(k) = dum*ng3d(k)/dt
      end if

!        nsubr(k)=0.
!        nsubs(k)=0.
!        nsubg(k)=0.

! update tendencies

!        nc3dten(k) = nc3dten(k)+nsubc(k)
         ni3dten(k) = ni3dten(k)+nsubi(k)
         ns3dten(k) = ns3dten(k)+nsubs(k)
         ng3dten(k) = ng3dten(k)+nsubg(k)
         nr3dten(k) = nr3dten(k)+nsubr(k)

#if (wrf_chem == 1)
      if( has_wetscav ) then
         evapprod(k) = - pre(k) - eprds(k) - eprdg(k) 
         rainprod(k) = pra(k) + prc(k) + psacws(k) + psacwg(k) + pgsacw(k) & 
                       + prai(k) + prci(k) + praci(k) + pracis(k)  &
                       + prds(k) + prdg(k)
      endif
#endif

         end if !!!!!! temperature

! switch ltrue to 1, since hydrometeors are present
         ltrue = 1

 200     continue

        end do

! initialize precip and snow rates
      precrt = 0.
      snowrt = 0.
! hm added 7/13/13
      snowprt = 0.
      grplprt = 0.

! if there are no hydrometeors, then skip to end of subroutine

        if (ltrue.eq.0) goto 400

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!.......................................................................
! calculate sedimenation
! the numerics here follow from reisner et al. (1998)
! fallout terms are calculated on split time steps to ensure numerical
! stability, i.e. courant# < 1

!.......................................................................

      nstep = 1

      do k = kte,kts,-1

        dumi(k) = qi3d(k)+qi3dten(k)*dt
        dumqs(k) = qni3d(k)+qni3dten(k)*dt
        dumr(k) = qr3d(k)+qr3dten(k)*dt
        dumfni(k) = ni3d(k)+ni3dten(k)*dt
        dumfns(k) = ns3d(k)+ns3dten(k)*dt
        dumfnr(k) = nr3d(k)+nr3dten(k)*dt
        dumc(k) = qc3d(k)+qc3dten(k)*dt
        dumfnc(k) = nc3d(k)+nc3dten(k)*dt
	dumg(k) = qg3d(k)+qg3dten(k)*dt
	dumfng(k) = ng3d(k)+ng3dten(k)*dt

! switch for constant droplet number
        if (iinum.eq.1) then
        dumfnc(k) = nc3d(k)
        end if

! get dummy lamda for sedimentation calculations

! make sure number concentrations are positive
      dumfni(k) = max(0.,dumfni(k))
      dumfns(k) = max(0.,dumfns(k))
      dumfnc(k) = max(0.,dumfnc(k))
      dumfnr(k) = max(0.,dumfnr(k))
      dumfng(k) = max(0.,dumfng(k))

!......................................................................
! cloud ice

      if (dumi(k).ge.qsmall) then
        dlami = (cons12*dumfni(k)/dumi(k))**(1./di)
        dlami=max(dlami,lammini)
        dlami=min(dlami,lammaxi)
      end if
!......................................................................
! rain

      if (dumr(k).ge.qsmall) then
        dlamr = (pi*rhow*dumfnr(k)/dumr(k))**(1./3.)
        dlamr=max(dlamr,lamminr)
        dlamr=min(dlamr,lammaxr)
      end if
!......................................................................
! cloud droplets

      if (dumc(k).ge.qsmall) then
         dum = pres(k)/(287.15*t3d(k))
         pgam(k)=0.0005714*(nc3d(k)/1.e6*dum)+0.2714
         pgam(k)=1./(pgam(k)**2)-1.
         pgam(k)=max(pgam(k),2.)
         pgam(k)=min(pgam(k),10.)

        dlamc = (cons26*dumfnc(k)*gamma(pgam(k)+4.)/(dumc(k)*gamma(pgam(k)+1.)))**(1./3.)
        lammin = (pgam(k)+1.)/60.e-6
        lammax = (pgam(k)+1.)/1.e-6
        dlamc=max(dlamc,lammin)
        dlamc=min(dlamc,lammax)
      end if
!......................................................................
! snow

      if (dumqs(k).ge.qsmall) then
        dlams = (cons1*dumfns(k)/ dumqs(k))**(1./ds)
        dlams=max(dlams,lammins)
        dlams=min(dlams,lammaxs)
      end if
!......................................................................
! graupel

      if (dumg(k).ge.qsmall) then
        dlamg = (cons2*dumfng(k)/ dumg(k))**(1./dg)
        dlamg=max(dlamg,lamming)
        dlamg=min(dlamg,lammaxg)
      end if

!......................................................................
! calculate number-weighted and mass-weighted terminal fall speeds

! cloud water

      if (dumc(k).ge.qsmall) then
      unc =  acn(k)*gamma(1.+bc+pgam(k))/ (dlamc**bc*gamma(pgam(k)+1.))
      umc = acn(k)*gamma(4.+bc+pgam(k))/  (dlamc**bc*gamma(pgam(k)+4.))
      else
      umc = 0.
      unc = 0.
      end if

      if (dumi(k).ge.qsmall) then
      uni =  ain(k)*cons27/dlami**bi
      umi = ain(k)*cons28/(dlami**bi)
      else
      umi = 0.
      uni = 0.
      end if

      if (dumr(k).ge.qsmall) then
      unr = arn(k)*cons6/dlamr**br
      umr = arn(k)*cons4/(dlamr**br)
      else
      umr = 0.
      unr = 0.
      end if

      if (dumqs(k).ge.qsmall) then
      ums = asn(k)*cons3/(dlams**bs)
      uns = asn(k)*cons5/dlams**bs
      else
      ums = 0.
      uns = 0.
      end if

      if (dumg(k).ge.qsmall) then
      umg = agn(k)*cons7/(dlamg**bg)
      ung = agn(k)*cons8/dlamg**bg
      else
      umg = 0.
      ung = 0.
      end if

! set realistic limits on fallspeed

! bug fix, 10/08/09
        dum=(rhosu/rho(k))**0.54
        ums=min(ums,1.2*dum)
        uns=min(uns,1.2*dum)
! fix 053011
! fix for correction by aa 4/6/11
        umi=min(umi,1.2*(rhosu/rho(k))**0.35)
        uni=min(uni,1.2*(rhosu/rho(k))**0.35)
        umr=min(umr,9.1*dum)
        unr=min(unr,9.1*dum)
        umg=min(umg,20.*dum)
        ung=min(ung,20.*dum)

      fr(k) = umr
      fi(k) = umi
      fni(k) = uni
      fs(k) = ums
      fns(k) = uns
      fnr(k) = unr
      fc(k) = umc
      fnc(k) = unc
      fg(k) = umg
      fng(k) = ung

! v3.3 modify fallspeed below level of precip

	if (k.le.kte-1) then
        if (fr(k).lt.1.e-10) then
	fr(k)=fr(k+1)
	end if
        if (fi(k).lt.1.e-10) then
	fi(k)=fi(k+1)
	end if
        if (fni(k).lt.1.e-10) then
	fni(k)=fni(k+1)
	end if
        if (fs(k).lt.1.e-10) then
	fs(k)=fs(k+1)
	end if
        if (fns(k).lt.1.e-10) then
	fns(k)=fns(k+1)
	end if
        if (fnr(k).lt.1.e-10) then
	fnr(k)=fnr(k+1)
	end if
        if (fc(k).lt.1.e-10) then
	fc(k)=fc(k+1)
	end if
        if (fnc(k).lt.1.e-10) then
	fnc(k)=fnc(k+1)
	end if
        if (fg(k).lt.1.e-10) then
	fg(k)=fg(k+1)
	end if
        if (fng(k).lt.1.e-10) then
	fng(k)=fng(k+1)
	end if
	end if ! k le kte-1

! calculate number of split time steps

      rgvm = max(fr(k),fi(k),fs(k),fc(k),fni(k),fnr(k),fns(k),fnc(k),fg(k),fng(k))
! vvt changed ifix -> int (generic function)
      nstep = max(int(rgvm*dt/dzq(k)+1.),nstep)

! multiply variables by rho
      dumr(k) = dumr(k)*rho(k)
      dumi(k) = dumi(k)*rho(k)
      dumfni(k) = dumfni(k)*rho(k)
      dumqs(k) = dumqs(k)*rho(k)
      dumfns(k) = dumfns(k)*rho(k)
      dumfnr(k) = dumfnr(k)*rho(k)
      dumc(k) = dumc(k)*rho(k)
      dumfnc(k) = dumfnc(k)*rho(k)
      dumg(k) = dumg(k)*rho(k)
      dumfng(k) = dumfng(k)*rho(k)

      end do

      do n = 1,nstep

      do k = kts,kte
      faloutr(k) = fr(k)*dumr(k)
      falouti(k) = fi(k)*dumi(k)
      faloutni(k) = fni(k)*dumfni(k)
      falouts(k) = fs(k)*dumqs(k)
      faloutns(k) = fns(k)*dumfns(k)
      faloutnr(k) = fnr(k)*dumfnr(k)
      faloutc(k) = fc(k)*dumc(k)
      faloutnc(k) = fnc(k)*dumfnc(k)
      faloutg(k) = fg(k)*dumg(k)
      faloutng(k) = fng(k)*dumfng(k)
      end do

! top of model

      k = kte
      faltndr = faloutr(k)/dzq(k)
      faltndi = falouti(k)/dzq(k)
      faltndni = faloutni(k)/dzq(k)
      faltnds = falouts(k)/dzq(k)
      faltndns = faloutns(k)/dzq(k)
      faltndnr = faloutnr(k)/dzq(k)
      faltndc = faloutc(k)/dzq(k)
      faltndnc = faloutnc(k)/dzq(k)
      faltndg = faloutg(k)/dzq(k)
      faltndng = faloutng(k)/dzq(k)
! add fallout terms to eulerian tendencies

      qrsten(k) = qrsten(k)-faltndr/nstep/rho(k)
      qisten(k) = qisten(k)-faltndi/nstep/rho(k)
      ni3dten(k) = ni3dten(k)-faltndni/nstep/rho(k)
      qnisten(k) = qnisten(k)-faltnds/nstep/rho(k)
      ns3dten(k) = ns3dten(k)-faltndns/nstep/rho(k)
      nr3dten(k) = nr3dten(k)-faltndnr/nstep/rho(k)
      qcsten(k) = qcsten(k)-faltndc/nstep/rho(k)
      nc3dten(k) = nc3dten(k)-faltndnc/nstep/rho(k)
      qgsten(k) = qgsten(k)-faltndg/nstep/rho(k)
      ng3dten(k) = ng3dten(k)-faltndng/nstep/rho(k)

      dumr(k) = dumr(k)-faltndr*dt/nstep
      dumi(k) = dumi(k)-faltndi*dt/nstep
      dumfni(k) = dumfni(k)-faltndni*dt/nstep
      dumqs(k) = dumqs(k)-faltnds*dt/nstep
      dumfns(k) = dumfns(k)-faltndns*dt/nstep
      dumfnr(k) = dumfnr(k)-faltndnr*dt/nstep
      dumc(k) = dumc(k)-faltndc*dt/nstep
      dumfnc(k) = dumfnc(k)-faltndnc*dt/nstep
      dumg(k) = dumg(k)-faltndg*dt/nstep
      dumfng(k) = dumfng(k)-faltndng*dt/nstep

      do k = kte-1,kts,-1
      faltndr = (faloutr(k+1)-faloutr(k))/dzq(k)
      faltndi = (falouti(k+1)-falouti(k))/dzq(k)
      faltndni = (faloutni(k+1)-faloutni(k))/dzq(k)
      faltnds = (falouts(k+1)-falouts(k))/dzq(k)
      faltndns = (faloutns(k+1)-faloutns(k))/dzq(k)
      faltndnr = (faloutnr(k+1)-faloutnr(k))/dzq(k)
      faltndc = (faloutc(k+1)-faloutc(k))/dzq(k)
      faltndnc = (faloutnc(k+1)-faloutnc(k))/dzq(k)
      faltndg = (faloutg(k+1)-faloutg(k))/dzq(k)
      faltndng = (faloutng(k+1)-faloutng(k))/dzq(k)

! add fallout terms to eulerian tendencies

      qrsten(k) = qrsten(k)+faltndr/nstep/rho(k)
      qisten(k) = qisten(k)+faltndi/nstep/rho(k)
      ni3dten(k) = ni3dten(k)+faltndni/nstep/rho(k)
      qnisten(k) = qnisten(k)+faltnds/nstep/rho(k)
      ns3dten(k) = ns3dten(k)+faltndns/nstep/rho(k)
      nr3dten(k) = nr3dten(k)+faltndnr/nstep/rho(k)
      qcsten(k) = qcsten(k)+faltndc/nstep/rho(k)
      nc3dten(k) = nc3dten(k)+faltndnc/nstep/rho(k)
      qgsten(k) = qgsten(k)+faltndg/nstep/rho(k)
      ng3dten(k) = ng3dten(k)+faltndng/nstep/rho(k)

      dumr(k) = dumr(k)+faltndr*dt/nstep
      dumi(k) = dumi(k)+faltndi*dt/nstep
      dumfni(k) = dumfni(k)+faltndni*dt/nstep
      dumqs(k) = dumqs(k)+faltnds*dt/nstep
      dumfns(k) = dumfns(k)+faltndns*dt/nstep
      dumfnr(k) = dumfnr(k)+faltndnr*dt/nstep
      dumc(k) = dumc(k)+faltndc*dt/nstep
      dumfnc(k) = dumfnc(k)+faltndnc*dt/nstep
      dumg(k) = dumg(k)+faltndg*dt/nstep
      dumfng(k) = dumfng(k)+faltndng*dt/nstep

! for wrf-chem, need precip rates (units of kg/m^2/s)
	  csed(k)=csed(k)+faloutc(k)/nstep
	  ised(k)=ised(k)+falouti(k)/nstep
	  ssed(k)=ssed(k)+falouts(k)/nstep
	  gsed(k)=gsed(k)+faloutg(k)/nstep
	  rsed(k)=rsed(k)+faloutr(k)/nstep
      end do

! get precipitation and snowfall accumulation during the time step
! factor of 1000 converts from m to mm, but division by density
! of liquid water cancels this factor of 1000

        precrt = precrt+(faloutr(kts)+faloutc(kts)+falouts(kts)+falouti(kts)+faloutg(kts))  &
                     *dt/nstep
        snowrt = snowrt+(falouts(kts)+falouti(kts)+faloutg(kts))*dt/nstep
! hm added 7/13/13
        snowprt = snowprt+(falouti(kts)+falouts(kts))*dt/nstep
        grplprt = grplprt+(faloutg(kts))*dt/nstep

      end do

        do k=kts,kte

! add on sedimentation tendencies for mixing ratio to rest of tendencies

        qr3dten(k)=qr3dten(k)+qrsten(k)
        qi3dten(k)=qi3dten(k)+qisten(k)
        qc3dten(k)=qc3dten(k)+qcsten(k)
        qg3dten(k)=qg3dten(k)+qgsten(k)
        qni3dten(k)=qni3dten(k)+qnisten(k)

! put all cloud ice in snow category if mean diameter exceeds 2 * dcs

!hm 4/7/09 bug fix
!        if (qi3d(k).ge.qsmall.and.t3d(k).lt.273.15) then
        if (qi3d(k).ge.qsmall.and.t3d(k).lt.273.15.and.lami(k).ge.1.e-10) then
        if (1./lami(k).ge.2.*dcs) then
           qni3dten(k) = qni3dten(k)+qi3d(k)/dt+ qi3dten(k)
           ns3dten(k) = ns3dten(k)+ni3d(k)/dt+   ni3dten(k)
           qi3dten(k) = -qi3d(k)/dt
           ni3dten(k) = -ni3d(k)/dt
        end if
        end if

! hm add tendencies here, then call sizeparameter
! to ensure consisitency between mixing ratio and number concentration

          qc3d(k)        = qc3d(k)+qc3dten(k)*dt
          qi3d(k)        = qi3d(k)+qi3dten(k)*dt
          qni3d(k)        = qni3d(k)+qni3dten(k)*dt
          qr3d(k)        = qr3d(k)+qr3dten(k)*dt
          nc3d(k)        = nc3d(k)+nc3dten(k)*dt
          ni3d(k)        = ni3d(k)+ni3dten(k)*dt
          ns3d(k)        = ns3d(k)+ns3dten(k)*dt
          nr3d(k)        = nr3d(k)+nr3dten(k)*dt

          if (igraup.eq.0) then
          qg3d(k)        = qg3d(k)+qg3dten(k)*dt
          ng3d(k)        = ng3d(k)+ng3dten(k)*dt
          end if

! add temperature and water vapor tendencies from microphysics
          t3d(k)         = t3d(k)+t3dten(k)*dt
          qv3d(k)        = qv3d(k)+qv3dten(k)*dt

! saturation vapor pressure and mixing ratio

! hm, add fix for low pressure, 5/12/10
            evs(k) = min(0.99*pres(k),polysvp(t3d(k),0))   ! pa
            eis(k) = min(0.99*pres(k),polysvp(t3d(k),1))   ! pa

! make sure ice saturation doesn't exceed water sat. near freezing

            if (eis(k).gt.evs(k)) eis(k) = evs(k)

            qvs(k) = ep_2*evs(k)/(pres(k)-evs(k))
            qvi(k) = ep_2*eis(k)/(pres(k)-eis(k))

            qvqvs(k) = qv3d(k)/qvs(k)
            qvqvsi(k) = qv3d(k)/qvi(k)

! at subsaturation, remove small amounts of cloud/precip water
! hm 7/9/09 change limit to 1.e-8

             if (qvqvs(k).lt.0.9) then
               if (qr3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qr3d(k)
                  t3d(k)=t3d(k)-qr3d(k)*xxlv(k)/cpm(k)
                  qr3d(k)=0.
               end if
               if (qc3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qc3d(k)
                  t3d(k)=t3d(k)-qc3d(k)*xxlv(k)/cpm(k)
                  qc3d(k)=0.
               end if
             end if

             if (qvqvsi(k).lt.0.9) then
               if (qi3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qi3d(k)
                  t3d(k)=t3d(k)-qi3d(k)*xxls(k)/cpm(k)
                  qi3d(k)=0.
               end if
               if (qni3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qni3d(k)
                  t3d(k)=t3d(k)-qni3d(k)*xxls(k)/cpm(k)
                  qni3d(k)=0.
               end if
               if (qg3d(k).lt.1.e-8) then
                  qv3d(k)=qv3d(k)+qg3d(k)
                  t3d(k)=t3d(k)-qg3d(k)*xxls(k)/cpm(k)
                  qg3d(k)=0.
               end if
             end if

!..................................................................
! if mixing ratio < qsmall set mixing ratio and number conc to zero

       if (qc3d(k).lt.qsmall) then
         qc3d(k) = 0.
         nc3d(k) = 0.
         effc(k) = 0.
       end if
       if (qr3d(k).lt.qsmall) then
         qr3d(k) = 0.
         nr3d(k) = 0.
         effr(k) = 0.
       end if
       if (qi3d(k).lt.qsmall) then
         qi3d(k) = 0.
         ni3d(k) = 0.
         effi(k) = 0.
       end if
       if (qni3d(k).lt.qsmall) then
         qni3d(k) = 0.
         ns3d(k) = 0.
         effs(k) = 0.
       end if
       if (qg3d(k).lt.qsmall) then
         qg3d(k) = 0.
         ng3d(k) = 0.
         effg(k) = 0.
       end if

!..................................
! if there is no cloud/precip water, then skip calculations

            if (qc3d(k).lt.qsmall.and.qi3d(k).lt.qsmall.and.qni3d(k).lt.qsmall &
                 .and.qr3d(k).lt.qsmall.and.qg3d(k).lt.qsmall) goto 500

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! calculate instantaneous processes

! add melting of cloud ice to form rain

        if (qi3d(k).ge.qsmall.and.t3d(k).ge.273.15) then
           qr3d(k) = qr3d(k)+qi3d(k)
           t3d(k) = t3d(k)-qi3d(k)*xlf(k)/cpm(k)
#if (wrf_chem == 1)
           tqimelt(k)=qi3d(k)/dt
#endif
           qi3d(k) = 0.
           nr3d(k) = nr3d(k)+ni3d(k)
           ni3d(k) = 0.
        end if

! ****sensitivity - no ice
        if (iliq.eq.1) goto 778

! homogeneous freezing of cloud water

        if (t3d(k).le.233.15.and.qc3d(k).ge.qsmall) then
           qi3d(k)=qi3d(k)+qc3d(k)
           t3d(k)=t3d(k)+qc3d(k)*xlf(k)/cpm(k)
           qc3d(k)=0.
           ni3d(k)=ni3d(k)+nc3d(k)
           nc3d(k)=0.
        end if

! homogeneous freezing of rain

        if (igraup.eq.0) then

        if (t3d(k).le.233.15.and.qr3d(k).ge.qsmall) then
           qg3d(k) = qg3d(k)+qr3d(k)
           t3d(k) = t3d(k)+qr3d(k)*xlf(k)/cpm(k)
           qr3d(k) = 0.
           ng3d(k) = ng3d(k)+ nr3d(k)
           nr3d(k) = 0.
        end if

        else if (igraup.eq.1) then

        if (t3d(k).le.233.15.and.qr3d(k).ge.qsmall) then
           qni3d(k) = qni3d(k)+qr3d(k)
           t3d(k) = t3d(k)+qr3d(k)*xlf(k)/cpm(k)
           qr3d(k) = 0.
           ns3d(k) = ns3d(k)+nr3d(k)
           nr3d(k) = 0.
        end if

        end if

 778    continue

! make sure number concentrations aren't negative

      ni3d(k) = max(0.,ni3d(k))
      ns3d(k) = max(0.,ns3d(k))
      nc3d(k) = max(0.,nc3d(k))
      nr3d(k) = max(0.,nr3d(k))
      ng3d(k) = max(0.,ng3d(k))

!......................................................................
! cloud ice

      if (qi3d(k).ge.qsmall) then
         lami(k) = (cons12*                 &
              ni3d(k)/qi3d(k))**(1./di)

! check for slope

! adjust vars

      if (lami(k).lt.lammini) then

      lami(k) = lammini

      n0i(k) = lami(k)**4*qi3d(k)/cons12

      ni3d(k) = n0i(k)/lami(k)
      else if (lami(k).gt.lammaxi) then
      lami(k) = lammaxi
      n0i(k) = lami(k)**4*qi3d(k)/cons12

      ni3d(k) = n0i(k)/lami(k)
      end if
      end if

!......................................................................
! rain

      if (qr3d(k).ge.qsmall) then
      lamr(k) = (pi*rhow*nr3d(k)/qr3d(k))**(1./3.)

! check for slope

! adjust vars

      if (lamr(k).lt.lamminr) then

      lamr(k) = lamminr

      n0rr(k) = lamr(k)**4*qr3d(k)/(pi*rhow)

      nr3d(k) = n0rr(k)/lamr(k)
      else if (lamr(k).gt.lammaxr) then
      lamr(k) = lammaxr
      n0rr(k) = lamr(k)**4*qr3d(k)/(pi*rhow)

      nr3d(k) = n0rr(k)/lamr(k)
      end if

      end if

!......................................................................
! cloud droplets

! martin et al. (1994) formula for pgam

      if (qc3d(k).ge.qsmall) then

         dum = pres(k)/(287.15*t3d(k))
         pgam(k)=0.0005714*(nc3d(k)/1.e6*dum)+0.2714
         pgam(k)=1./(pgam(k)**2)-1.
         pgam(k)=max(pgam(k),2.)
         pgam(k)=min(pgam(k),10.)

! calculate lamc

      lamc(k) = (cons26*nc3d(k)*gamma(pgam(k)+4.)/   &
                 (qc3d(k)*gamma(pgam(k)+1.)))**(1./3.)

! lammin, 60 micron diameter
! lammax, 1 micron

      lammin = (pgam(k)+1.)/60.e-6
      lammax = (pgam(k)+1.)/1.e-6

      if (lamc(k).lt.lammin) then
      lamc(k) = lammin
      nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+              &
                log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26

      else if (lamc(k).gt.lammax) then
      lamc(k) = lammax
      nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+              &
                log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26

      end if

      end if

!......................................................................
! snow

      if (qni3d(k).ge.qsmall) then
      lams(k) = (cons1*ns3d(k)/qni3d(k))**(1./ds)

! check for slope

! adjust vars

      if (lams(k).lt.lammins) then
      lams(k) = lammins
      n0s(k) = lams(k)**4*qni3d(k)/cons1

      ns3d(k) = n0s(k)/lams(k)

      else if (lams(k).gt.lammaxs) then

      lams(k) = lammaxs
      n0s(k) = lams(k)**4*qni3d(k)/cons1
      ns3d(k) = n0s(k)/lams(k)
      end if

      end if

!......................................................................
! graupel

      if (qg3d(k).ge.qsmall) then
      lamg(k) = (cons2*ng3d(k)/qg3d(k))**(1./dg)

! check for slope

! adjust vars

      if (lamg(k).lt.lamming) then
      lamg(k) = lamming
      n0g(k) = lamg(k)**4*qg3d(k)/cons2

      ng3d(k) = n0g(k)/lamg(k)

      else if (lamg(k).gt.lammaxg) then

      lamg(k) = lammaxg
      n0g(k) = lamg(k)**4*qg3d(k)/cons2

      ng3d(k) = n0g(k)/lamg(k)
      end if

      end if

 500  continue

! calculate effective radius

      if (qi3d(k).ge.qsmall) then
         effi(k) = 3./lami(k)/2.*1.e6
      else
         effi(k) = 25.
      end if

      if (qni3d(k).ge.qsmall) then
         effs(k) = 3./lams(k)/2.*1.e6
      else
         effs(k) = 25.
      end if

      if (qr3d(k).ge.qsmall) then
         effr(k) = 3./lamr(k)/2.*1.e6
      else
         effr(k) = 25.
      end if

      if (qc3d(k).ge.qsmall) then
      effc(k) = gamma(pgam(k)+4.)/                        &
             gamma(pgam(k)+3.)/lamc(k)/2.*1.e6
      else
      effc(k) = 25.
      end if

      if (qg3d(k).ge.qsmall) then
         effg(k) = 3./lamg(k)/2.*1.e6
      else
         effg(k) = 25.
      end if

! hm add 1/10/06, add upper bound on ice number, this is needed
! to prevent very large ice number due to homogeneous freezing
! of droplets, especially when inum = 1, set max at 10 cm-3
!          ni3d(k) = min(ni3d(k),10.e6/rho(k))
! hm, 12/28/12, lower maximum ice concentration to address problem
! of excessive and persistent anvil
! note: this may change/reduce sensitivity to aerosol/ccn concentration
          ni3d(k) = min(ni3d(k),0.3e6/rho(k))

! add bound on droplet number - cannot exceed aerosol concentration
          if (iinum.eq.0.and.iact.eq.2) then
          nc3d(k) = min(nc3d(k),(nanew1+nanew2)/rho(k))
          end if
! switch for constant droplet number
          if (iinum.eq.1) then 
! change ndcnst from cm-3 to kg-1
             nc3d(k) = ndcnst*1.e6/rho(k)
          end if

      end do !!! k loop

 400         continue

! all done !!!!!!!!!!!
      return
      end subroutine morr_two_moment_micro

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      real function polysvp (t,type)

!-------------------------------------------

!  compute saturation vapor pressure

!  polysvp returned in units of pa.
!  t is input in units of k.
!  type refers to saturation with respect to liquid (0) or ice (1)

! replace goff-gratch with faster formulation from flatau et al. 1992, table 4 (right-hand column)

      implicit none

      real dum
      real t
      integer type
! ice
      real a0i,a1i,a2i,a3i,a4i,a5i,a6i,a7i,a8i 
      data a0i,a1i,a2i,a3i,a4i,a5i,a6i,a7i,a8i /&
	6.11147274, 0.503160820, 0.188439774e-1, &
        0.420895665e-3, 0.615021634e-5,0.602588177e-7, &
        0.385852041e-9, 0.146898966e-11, 0.252751365e-14/	

! liquid
      real a0,a1,a2,a3,a4,a5,a6,a7,a8 

! v1.7
      data a0,a1,a2,a3,a4,a5,a6,a7,a8 /&
	6.11239921, 0.443987641, 0.142986287e-1, &
        0.264847430e-3, 0.302950461e-5, 0.206739458e-7, &
        0.640689451e-10,-0.952447341e-13,-0.976195544e-15/
      real dt

! ice

      if (type.eq.1) then

!         polysvp = 10.**(-9.09718*(273.16/t-1.)-3.56654*                &
!          log10(273.16/t)+0.876793*(1.-t/273.16)+						&
!          log10(6.1071))*100.
! hm 11/16/20, use goff-gratch for t < 195.8 k and flatau et al. equal or above 195.8 k
      if (t.ge.195.8) then
         dt=t-273.15
         polysvp = a0i + dt*(a1i+dt*(a2i+dt*(a3i+dt*(a4i+dt*(a5i+dt*(a6i+dt*(a7i+a8i*dt))))))) 
         polysvp = polysvp*100.
      else

         polysvp = 10.**(-9.09718*(273.16/t-1.)-3.56654* &
          alog10(273.16/t)+0.876793*(1.-t/273.16)+ &
          alog10(6.1071))*100.

      end if

      end if

! liquid

      if (type.eq.0) then

!         polysvp = 10.**(-7.90298*(373.16/t-1.)+                        &
!             5.02808*log10(373.16/t)-									&
!             1.3816e-7*(10**(11.344*(1.-t/373.16))-1.)+				&
!             8.1328e-3*(10**(-3.49149*(373.16/t-1.))-1.)+				&
!             log10(1013.246))*100.
! hm 11/16/20, use goff-gratch for t < 202.0 k and flatau et al. equal or above 202.0 k
      if (t.ge.202.0) then
         dt = t-273.15
         polysvp = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))))
         polysvp = polysvp*100.
      else

! note: uncertain below -70 c, but produces physical values (non-negative) unlike flatau
         polysvp = 10.**(-7.90298*(373.16/t-1.)+ &
             5.02808*alog10(373.16/t)- &
             1.3816e-7*(10**(11.344*(1.-t/373.16))-1.)+ &
             8.1328e-3*(10**(-3.49149*(373.16/t-1.))-1.)+ &
             alog10(1013.246))*100.                      
      end if

      end if


      end function polysvp

!------------------------------------------------------------------------------

      real function gamma(x)
!----------------------------------------------------------------------
!
! this routine calculates the gamma function for a real argument x.
!   computation is based on an algorithm outlined in reference 1.
!   the program uses rational functions that approximate the gamma
!   function to at least 20 significant decimal digits.  coefficients
!   for the approximation over the interval (1,2) are unpublished.
!   those for the approximation for x .ge. 12 are from reference 2.
!   the accuracy achieved depends on the arithmetic system, the
!   compiler, the intrinsic functions, and proper selection of the
!   machine-dependent constants.
!
!
!*******************************************************************
!*******************************************************************
!
! explanation of machine-dependent constants
!
! beta   - radix for the floating-point representation
! maxexp - the smallest positive power of beta that overflows
! xbig   - the largest argument for which gamma(x) is representable
!          in the machine, i.e., the solution to the equation
!                  gamma(xbig) = beta**maxexp
! xinf   - the largest machine representable floating-point number;
!          approximately beta**maxexp
! eps    - the smallest positive floating-point number such that
!          1.0+eps .gt. 1.0
! xminin - the smallest positive floating-point number such that
!          1/xminin is machine representable
!
!     approximate values for some important machines are:
!
!                            beta       maxexp        xbig
!
! cray-1         (s.p.)        2         8191        966.961
! cyber 180/855
!   under nos    (s.p.)        2         1070        177.803
! ieee (ibm/xt,
!   sun, etc.)   (s.p.)        2          128        35.040
! ieee (ibm/xt,
!   sun, etc.)   (d.p.)        2         1024        171.624
! ibm 3033       (d.p.)       16           63        57.574
! vax d-format   (d.p.)        2          127        34.844
! vax g-format   (d.p.)        2         1023        171.489
!
!                            xinf         eps        xminin
!
! cray-1         (s.p.)   5.45e+2465   7.11e-15    1.84e-2466
! cyber 180/855
!   under nos    (s.p.)   1.26e+322    3.55e-15    3.14e-294
! ieee (ibm/xt,
!   sun, etc.)   (s.p.)   3.40e+38     1.19e-7     1.18e-38
! ieee (ibm/xt,
!   sun, etc.)   (d.p.)   1.79d+308    2.22d-16    2.23d-308
! ibm 3033       (d.p.)   7.23d+75     2.22d-16    1.39d-76
! vax d-format   (d.p.)   1.70d+38     1.39d-17    5.88d-39
! vax g-format   (d.p.)   8.98d+307    1.11d-16    1.12d-308
!
!*******************************************************************
!*******************************************************************
!
! error returns
!
!  the program returns the value xinf for singularities or
!     when overflow would occur.  the computation is believed
!     to be free of underflow and overflow.
!
!
!  intrinsic functions required are:
!
!     int, dble, exp, log, real, sin
!
!
! references:  an overview of software development for special
!              functions   w. j. cody, lecture notes in mathematics,
!              506, numerical analysis dundee, 1975, g. a. watson
!              (ed.), springer verlag, berlin, 1976.
!
!              computer approximations, hart, et. al., wiley and
!              sons, new york, 1968.
!
!  latest modification: october 12, 1989
!
!  authors: w. j. cody and l. stoltz
!           applied mathematics division
!           argonne national laboratory
!           argonne, il 60439
!
!----------------------------------------------------------------------
      implicit none
      integer i,n
      logical parity
      real                                                          &
          conv,eps,fact,half,one,res,sum,twelve,                    &
          two,x,xbig,xden,xinf,xminin,xnum,y,y1,ysq,z,zero
      real, dimension(7) :: c
      real, dimension(8) :: p
      real, dimension(8) :: q
!----------------------------------------------------------------------
!  mathematical constants
!----------------------------------------------------------------------
      data one,half,twelve,two,zero/1.0e0,0.5e0,12.0e0,2.0e0,0.0e0/


!----------------------------------------------------------------------
!  machine dependent parameters
!----------------------------------------------------------------------
      data xbig,xminin,eps/35.040e0,1.18e-38,1.19e-7/,xinf/3.4e38/
!----------------------------------------------------------------------
!  numerator and denominator coefficients for rational minimax
!     approximation over (1,2).
!----------------------------------------------------------------------
      data p/-1.71618513886549492533811e+0,2.47656508055759199108314e+1,  &
             -3.79804256470945635097577e+2,6.29331155312818442661052e+2,  &
             8.66966202790413211295064e+2,-3.14512729688483675254357e+4,  &
             -3.61444134186911729807069e+4,6.64561438202405440627855e+4/
      data q/-3.08402300119738975254353e+1,3.15350626979604161529144e+2,  &
             -1.01515636749021914166146e+3,-3.10777167157231109440444e+3, &
              2.25381184209801510330112e+4,4.75584627752788110767815e+3,  &
            -1.34659959864969306392456e+5,-1.15132259675553483497211e+5/
!----------------------------------------------------------------------
!  coefficients for minimax approximation over (12, inf).
!----------------------------------------------------------------------
      data c/-1.910444077728e-03,8.4171387781295e-04,                      &
           -5.952379913043012e-04,7.93650793500350248e-04,				   &
           -2.777777777777681622553e-03,8.333333333333333331554247e-02,	   &
            5.7083835261e-03/
!----------------------------------------------------------------------
!  statement functions for conversion between integer and float
!----------------------------------------------------------------------
      conv(i) = real(i)
      parity=.false.
      fact=one
      n=0
      y=x
      if(y.le.zero)then
!----------------------------------------------------------------------
!  argument is negative
!----------------------------------------------------------------------
        y=-x
        y1=aint(y)
        res=y-y1
        if(res.ne.zero)then
          if(y1.ne.aint(y1*half)*two)parity=.true.
          fact=-pi/sin(pi*res)
          y=y+one
        else
          res=xinf
          goto 900
        endif
      endif
!----------------------------------------------------------------------
!  argument is positive
!----------------------------------------------------------------------
      if(y.lt.eps)then
!----------------------------------------------------------------------
!  argument .lt. eps
!----------------------------------------------------------------------
        if(y.ge.xminin)then
          res=one/y
        else
          res=xinf
          goto 900
        endif
      elseif(y.lt.twelve)then
        y1=y
        if(y.lt.one)then
!----------------------------------------------------------------------
!  0.0 .lt. argument .lt. 1.0
!----------------------------------------------------------------------
          z=y
          y=y+one
        else
!----------------------------------------------------------------------
!  1.0 .lt. argument .lt. 12.0, reduce argument if necessary
!----------------------------------------------------------------------
          n=int(y)-1
          y=y-conv(n)
          z=y-one
        endif
!----------------------------------------------------------------------
!  evaluate approximation for 1.0 .lt. argument .lt. 2.0
!----------------------------------------------------------------------
        xnum=zero
        xden=one
        do i=1,8
          xnum=(xnum+p(i))*z
          xden=xden*z+q(i)
        end do
        res=xnum/xden+one
        if(y1.lt.y)then
!----------------------------------------------------------------------
!  adjust result for case  0.0 .lt. argument .lt. 1.0
!----------------------------------------------------------------------
          res=res/y1
        elseif(y1.gt.y)then
!----------------------------------------------------------------------
!  adjust result for case  2.0 .lt. argument .lt. 12.0
!----------------------------------------------------------------------
          do i=1,n
            res=res*y
            y=y+one
          end do
        endif
      else
!----------------------------------------------------------------------
!  evaluate for argument .ge. 12.0,
!----------------------------------------------------------------------
        if(y.le.xbig)then
          ysq=y*y
          sum=c(7)
          do i=1,6
            sum=sum/ysq+c(i)
          end do
          sum=sum/y-y+xxx
          sum=sum+(y-half)*log(y)
          res=exp(sum)
        else
          res=xinf
          goto 900
        endif
      endif
!----------------------------------------------------------------------
!  final adjustments and return
!----------------------------------------------------------------------
      if(parity)res=-res
      if(fact.ne.one)res=fact/res
  900 gamma=res
      return
! ---------- last line of gamma ----------
      end function gamma


      real function derf1(x)
      implicit none
      real x
      real, dimension(0 : 64) :: a, b
      real w,t,y
      integer k,i
      data a/                                                 &
         0.00000000005958930743e0, -0.00000000113739022964e0, &
         0.00000001466005199839e0, -0.00000016350354461960e0, &
         0.00000164610044809620e0, -0.00001492559551950604e0, &
         0.00012055331122299265e0, -0.00085483269811296660e0, &
         0.00522397762482322257e0, -0.02686617064507733420e0, &
         0.11283791670954881569e0, -0.37612638903183748117e0, &
         1.12837916709551257377e0,	                          &
         0.00000000002372510631e0, -0.00000000045493253732e0, &
         0.00000000590362766598e0, -0.00000006642090827576e0, &
         0.00000067595634268133e0, -0.00000621188515924000e0, &
         0.00005103883009709690e0, -0.00037015410692956173e0, &
         0.00233307631218880978e0, -0.01254988477182192210e0, &
         0.05657061146827041994e0, -0.21379664776456006580e0, &
         0.84270079294971486929e0,							  &
         0.00000000000949905026e0, -0.00000000018310229805e0, &
         0.00000000239463074000e0, -0.00000002721444369609e0, &
         0.00000028045522331686e0, -0.00000261830022482897e0, &
         0.00002195455056768781e0, -0.00016358986921372656e0, &
         0.00107052153564110318e0, -0.00608284718113590151e0, &
         0.02986978465246258244e0, -0.13055593046562267625e0, &
         0.67493323603965504676e0, 							  &
         0.00000000000382722073e0, -0.00000000007421598602e0, &
         0.00000000097930574080e0, -0.00000001126008898854e0, &
         0.00000011775134830784e0, -0.00000111992758382650e0, &
         0.00000962023443095201e0, -0.00007404402135070773e0, &
         0.00050689993654144881e0, -0.00307553051439272889e0, &
         0.01668977892553165586e0, -0.08548534594781312114e0, &
         0.56909076642393639985e0,							  &
         0.00000000000155296588e0, -0.00000000003032205868e0, &
         0.00000000040424830707e0, -0.00000000471135111493e0, &
         0.00000005011915876293e0, -0.00000048722516178974e0, &
         0.00000430683284629395e0, -0.00003445026145385764e0, &
         0.00024879276133931664e0, -0.00162940941748079288e0, &
         0.00988786373932350462e0, -0.05962426839442303805e0, &
         0.49766113250947636708e0 /
      data (b(i), i = 0, 12) /                                  &
         -0.00000000029734388465e0,  0.00000000269776334046e0, 	&
         -0.00000000640788827665e0, -0.00000001667820132100e0,  &
         -0.00000021854388148686e0,  0.00000266246030457984e0, 	&
          0.00001612722157047886e0, -0.00025616361025506629e0, 	&
          0.00015380842432375365e0,  0.00815533022524927908e0, 	&
         -0.01402283663896319337e0, -0.19746892495383021487e0,  &
          0.71511720328842845913e0 /
      data (b(i), i = 13, 25) /                                 &
         -0.00000000001951073787e0, -0.00000000032302692214e0,  &
          0.00000000522461866919e0,  0.00000000342940918551e0, 	&
         -0.00000035772874310272e0,  0.00000019999935792654e0, 	&
          0.00002687044575042908e0, -0.00011843240273775776e0, 	&
         -0.00080991728956032271e0,  0.00661062970502241174e0, 	&
          0.00909530922354827295e0, -0.20160072778491013140e0, 	&
          0.51169696718727644908e0 /
      data (b(i), i = 26, 38) /                                 &
         0.00000000003147682272e0, -0.00000000048465972408e0,   &
         0.00000000063675740242e0,  0.00000003377623323271e0, 	&
        -0.00000015451139637086e0, -0.00000203340624738438e0, 	&
         0.00001947204525295057e0,  0.00002854147231653228e0, 	&
        -0.00101565063152200272e0,  0.00271187003520095655e0, 	&
         0.02328095035422810727e0, -0.16725021123116877197e0, 	&
         0.32490054966649436974e0 /
      data (b(i), i = 39, 51) /                                 &
         0.00000000002319363370e0, -0.00000000006303206648e0,   &
        -0.00000000264888267434e0,  0.00000002050708040581e0, 	&
         0.00000011371857327578e0, -0.00000211211337219663e0, 	&
         0.00000368797328322935e0,  0.00009823686253424796e0, 	&
        -0.00065860243990455368e0, -0.00075285814895230877e0, 	&
         0.02585434424202960464e0, -0.11637092784486193258e0, 	&
         0.18267336775296612024e0 /
      data (b(i), i = 52, 64) /                                 &
        -0.00000000000367789363e0,  0.00000000020876046746e0, 	&
        -0.00000000193319027226e0, -0.00000000435953392472e0, 	&
         0.00000018006992266137e0, -0.00000078441223763969e0, 	&
        -0.00000675407647949153e0,  0.00008428418334440096e0, 	&
        -0.00017604388937031815e0, -0.00239729611435071610e0, 	&
         0.02064129023876022970e0, -0.06905562880005864105e0,   &
         0.09084526782065478489e0 /
      w = abs(x)
      if (w .lt. 2.2d0) then
          t = w * w
          k = int(t)
          t = t - k
          k = k * 13
          y = ((((((((((((a(k) * t + a(k + 1)) * t +              &
              a(k + 2)) * t + a(k + 3)) * t + a(k + 4)) * t +     &
              a(k + 5)) * t + a(k + 6)) * t + a(k + 7)) * t +     &
              a(k + 8)) * t + a(k + 9)) * t + a(k + 10)) * t + 	  &
              a(k + 11)) * t + a(k + 12)) * w
      else if (w .lt. 6.9d0) then
          k = int(w)
          t = w - k
          k = 13 * (k - 2)
          y = (((((((((((b(k) * t + b(k + 1)) * t +               &
              b(k + 2)) * t + b(k + 3)) * t + b(k + 4)) * t + 	  &
              b(k + 5)) * t + b(k + 6)) * t + b(k + 7)) * t + 	  &
              b(k + 8)) * t + b(k + 9)) * t + b(k + 10)) * t + 	  &
              b(k + 11)) * t + b(k + 12)
          y = y * y
          y = y * y
          y = y * y
          y = 1 - y * y
      else
          y = 1
      end if
      if (x .lt. 0) y = -y
      derf1 = y
      end function derf1

!+---+-----------------------------------------------------------------+

      subroutine refl10cm_hm (qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d, &
                      t1d, p1d, dbz, kts, kte, ii, jj)

      implicit none

!..sub arguments
      integer, intent(in):: kts, kte, ii, jj
      real, dimension(kts:kte), intent(in)::                            &
                      qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d, t1d, p1d
      real, dimension(kts:kte), intent(inout):: dbz

!..local variables
      real, dimension(kts:kte):: temp, pres, qv, rho
      real, dimension(kts:kte):: rr, nr, rs, ns, rg, ng

      double precision, dimension(kts:kte):: ilamr, ilamg, ilams
      double precision, dimension(kts:kte):: n0_r, n0_g, n0_s
      double precision:: lamr, lamg, lams
      logical, dimension(kts:kte):: l_qr, l_qs, l_qg

      real, dimension(kts:kte):: ze_rain, ze_snow, ze_graupel
      double precision:: fmelt_s, fmelt_g
      double precision:: cback, x, eta, f_d

      integer:: i, k, k_0, kbot, n
      logical:: melti

!+---+

      do k = kts, kte
         dbz(k) = -35.0
      enddo

!+---+-----------------------------------------------------------------+
!..put column of data into local arrays.
!+---+-----------------------------------------------------------------+
      do k = kts, kte
         temp(k) = t1d(k)
         qv(k) = max(1.e-10, qv1d(k))
         pres(k) = p1d(k)
         rho(k) = 0.622*pres(k)/(r*temp(k)*(qv(k)+0.622))

         if (qr1d(k) .gt. 1.e-9) then
            rr(k) = qr1d(k)*rho(k)
            nr(k) = nr1d(k)*rho(k)
            lamr = (xam_r*xcrg(3)*xorg2*nr(k)/rr(k))**xobmr
            ilamr(k) = 1./lamr
            n0_r(k) = nr(k)*xorg2*lamr**xcre(2)
            l_qr(k) = .true.
         else
            rr(k) = 1.e-12
            nr(k) = 1.e-12
            l_qr(k) = .false.
         endif

         if (qs1d(k) .gt. 1.e-9) then
            rs(k) = qs1d(k)*rho(k)
            ns(k) = ns1d(k)*rho(k)
            lams = (xam_s*xcsg(3)*xosg2*ns(k)/rs(k))**xobms
            ilams(k) = 1./lams
            n0_s(k) = ns(k)*xosg2*lams**xcse(2)
            l_qs(k) = .true.
         else
            rs(k) = 1.e-12
            ns(k) = 1.e-12
            l_qs(k) = .false.
         endif

         if (qg1d(k) .gt. 1.e-9) then
            rg(k) = qg1d(k)*rho(k)
            ng(k) = ng1d(k)*rho(k)
            lamg = (xam_g*xcgg(3)*xogg2*ng(k)/rg(k))**xobmg
            ilamg(k) = 1./lamg
            n0_g(k) = ng(k)*xogg2*lamg**xcge(2)
            l_qg(k) = .true.
         else
            rg(k) = 1.e-12
            ng(k) = 1.e-12
            l_qg(k) = .false.
         endif
      enddo

!+---+-----------------------------------------------------------------+
!..locate k-level of start of melting (k_0 is level above).
!+---+-----------------------------------------------------------------+
      melti = .false.
      k_0 = kts
      do k = kte-1, kts, -1
         if ( (temp(k).gt.273.15) .and. l_qr(k)                         &
                                  .and. (l_qs(k+1).or.l_qg(k+1)) ) then
            k_0 = max(k+1, k_0)
            melti=.true.
            goto 195
         endif
      enddo
 195  continue

!+---+-----------------------------------------------------------------+
!..assume rayleigh approximation at 10 cm wavelength. rain (all temps)
!.. and non-water-coated snow and graupel when below freezing are
!.. simple. integrations of m(d)*m(d)*n(d)*dd.
!+---+-----------------------------------------------------------------+

      do k = kts, kte
         ze_rain(k) = 1.e-22
         ze_snow(k) = 1.e-22
         ze_graupel(k) = 1.e-22
         if (l_qr(k)) ze_rain(k) = n0_r(k)*xcrg(4)*ilamr(k)**xcre(4)
         if (l_qs(k)) ze_snow(k) = (0.176/0.93) * (6.0/pi)*(6.0/pi)     &
                                 * (xam_s/900.0)*(xam_s/900.0)          &
                                 * n0_s(k)*xcsg(4)*ilams(k)**xcse(4)
         if (l_qg(k)) ze_graupel(k) = (0.176/0.93) * (6.0/pi)*(6.0/pi)  &
                                    * (xam_g/900.0)*(xam_g/900.0)       &
                                    * n0_g(k)*xcgg(4)*ilamg(k)**xcge(4)
      enddo

!+---+-----------------------------------------------------------------+
!..special case of melting ice (snow/graupel) particles.  assume the
!.. ice is surrounded by the liquid water.  fraction of meltwater is
!.. extremely simple based on amount found above the melting level.
!.. uses code from uli blahak (rayleigh_soak_wetgraupel and supporting
!.. routines).
!+---+-----------------------------------------------------------------+

      if (melti .and. k_0.ge.kts+1) then
       do k = k_0-1, kts, -1

!..reflectivity contributed by melting snow
          if (l_qs(k) .and. l_qs(k_0) ) then
           fmelt_s = max(0.005d0, min(1.0d0-rs(k)/rs(k_0), 0.99d0))
           eta = 0.d0
           lams = 1./ilams(k)
           do n = 1, nrbins
              x = xam_s * xxds(n)**xbm_s
              call rayleigh_soak_wetgraupel (x,dble(xocms),dble(xobms), &
                    fmelt_s, melt_outside_s, m_w_0, m_i_0, lamda_radar, &
                    cback, mixingrulestring_s, matrixstring_s,          &
                    inclusionstring_s, hoststring_s,                    &
                    hostmatrixstring_s, hostinclusionstring_s)
              f_d = n0_s(k)*xxds(n)**xmu_s * dexp(-lams*xxds(n))
              eta = eta + f_d * cback * simpson(n) * xdts(n)
           enddo
           ze_snow(k) = sngl(lamda4 / (pi5 * k_w) * eta)
          endif


!..reflectivity contributed by melting graupel

          if (l_qg(k) .and. l_qg(k_0) ) then
           fmelt_g = max(0.005d0, min(1.0d0-rg(k)/rg(k_0), 0.99d0))
           eta = 0.d0
           lamg = 1./ilamg(k)
           do n = 1, nrbins
              x = xam_g * xxdg(n)**xbm_g
              call rayleigh_soak_wetgraupel (x,dble(xocmg),dble(xobmg), &
                    fmelt_g, melt_outside_g, m_w_0, m_i_0, lamda_radar, &
                    cback, mixingrulestring_g, matrixstring_g,          &
                    inclusionstring_g, hoststring_g,                    &
                    hostmatrixstring_g, hostinclusionstring_g)
              f_d = n0_g(k)*xxdg(n)**xmu_g * dexp(-lamg*xxdg(n))
              eta = eta + f_d * cback * simpson(n) * xdtg(n)
           enddo
           ze_graupel(k) = sngl(lamda4 / (pi5 * k_w) * eta)
          endif

       enddo
      endif

      do k = kte, kts, -1
         dbz(k) = 10.*log10((ze_rain(k)+ze_snow(k)+ze_graupel(k))*1.d18)
      enddo


      end subroutine refl10cm_hm

!+---+-----------------------------------------------------------------+

end module module_mp_morr_two_moment
!+---+-----------------------------------------------------------------+
!+---+-----------------------------------------------------------------+
