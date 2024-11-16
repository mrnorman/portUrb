
module module_mp_morr_two_moment
  use module_wrf_error
  use module_mp_radar
  use module_model_constants, only: cp, g, r => r_d, rv => r_v, ep_2
  implicit none
  real, private, parameter :: pi = 3.1415926535897932384626434
  real, private, parameter :: xxx = 0.9189385332046727417803297
  public  ::  mp_morr_two_moment
  public  ::  polysvp
  private :: gamma
  private :: morr_two_moment_micro
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! switches for microphysics scheme
  ! iact = 1, use power-law ccn spectra, nccn = cs^k
  ! iact = 2, use lognormal aerosol size dist to derive ccn spectra
  ! iact = 3, activation calculated in module_mixactivate
  integer, private ::  iact
  ! inum = 0, predict droplet concentration
  ! inum = 1, assume constant droplet concentration   
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
  real, private :: ai,ac,as,ar,ag    ! 'a' parameter in fallspeed-diam relationship
  real, private :: bi,bc,bs,br,bg    ! 'b' parameter in fallspeed-diam relationship
  real, private :: rhosu             ! standard air density at 850 mb
  real, private :: rhow              ! density of liquid water
  real, private :: rhoi              ! bulk density of cloud ice
  real, private :: rhosn             ! bulk density of snow
  real, private :: rhog              ! bulk density of graupel
  real, private :: aimm              ! parameter in bigg immersion freezing
  real, private :: bimm              ! parameter in bigg immersion freezing
  real, private :: ecr               ! collection efficiency between droplets/rain and snow/rain
  real, private :: dcs               ! threshold size for cloud ice autoconversion
  real, private :: mi0               ! initial size of nucleated crystal
  real, private :: mg0               ! mass of embryo graupel
  real, private :: f1s               ! ventilation parameter for snow
  real, private :: f2s               ! ventilation parameter for snow
  real, private :: f1r               ! ventilation parameter for rain
  real, private :: f2r               ! ventilation parameter for rain
  real, private :: qsmall            ! smallest allowed hydrometeor mixing ratio
  real, private :: ci,di,cs,ds,cg,dg ! size distribution parameters for cloud ice, snow, graupel
  real, private :: eii               ! collection efficiency, ice-ice collisions
  real, private :: eci               ! collection efficiency, ice-droplet collisions
  real, private :: rin               ! radius of contact nuclei (m)
  real, private :: cpw               ! specific heat of liquid water
  real, private :: c1                ! 'c' in nccn = cs^k (cm-3)
  real, private :: k1                ! 'k' in nccn = cs^k
  real, private :: mw                ! molecular weight water (kg/mol)
  real, private :: osm               ! osmotic coefficient
  real, private :: vi                ! number of ion dissociated in solution
  real, private :: epsm              ! aerosol soluble fraction
  real, private :: rhoa              ! aerosol bulk density (kg/m3)
  real, private :: map               ! molecular weight aerosol (kg/mol)
  real, private :: ma                ! molecular weight of 'air' (kg/mol)
  real, private :: rr                ! universal gas constant
  real, private :: bact              ! activation parameter
  real, private :: rm1               ! geometric mean radius, mode 1 (m)
  real, private :: rm2               ! geometric mean radius, mode 2 (m)
  real, private :: nanew1            ! total aerosol concentration, mode 1 (m^-3)
  real, private :: nanew2            ! total aerosol concentration, mode 2 (m^-3)
  real, private :: sig1              ! standard deviation of aerosol s.d., mode 1
  real, private :: sig2              ! standard deviation of aerosol s.d., mode 2
  real, private :: f11               ! correction factor for activation, mode 1
  real, private :: f12               ! correction factor for activation, mode 1
  real, private :: f21               ! correction factor for activation, mode 2
  real, private :: f22               ! correction factor for activation, mode 2     
  real, private :: mmult             ! mass of splintered ice particle
  real, private :: lammaxi,lammini,lammaxr,lamminr,lammaxs,lammins,lammaxg,lamming
  real, private :: cons1,cons2,cons3,cons4,cons5,cons6,cons7,cons8,cons9,cons10
  real, private :: cons11,cons12,cons13,cons14,cons15,cons16,cons17,cons18,cons19,cons20
  real, private :: cons21,cons22,cons23,cons24,cons25,cons26,cons27,cons28,cons29,cons30
  real, private :: cons31,cons32,cons33,cons34,cons35,cons36,cons37,cons38,cons39,cons40
  real, private :: cons41
  interface pow
    module procedure pow_rr
    module procedure pow_ri
  end interface

contains

  subroutine morr_two_moment_init(morr_rimed_ice) bind(c,name="morr_two_moment_init") ! ras  
    ! this subroutine initializes all physical constants amnd parameters 
    ! needed by the microphysics scheme.
    ! needs to be called at first time step, prior to call to main microphysics interface
    implicit none
    integer, intent(in):: morr_rimed_ice ! ras  
    integer n,i
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
    ! switch for hail/graupel
    ! ihail = 0, dense precipitating ice is graupel
    ! ihail = 1, dense precipitating ice is hail
    ! note ---> recommend ihail = 1 for continental deep convection
    !ihail = 0 !changed to namelist option (morr_rimed_ice) by ras
    ! check if namelist option is feasible, otherwise default to graupel - ras
    if (morr_rimed_ice == 1) then
       ihail = 1
    else
       ihail = 0
    endif
    ! fallspeed parameters (v=ad^b)
    ai = 700.
    ac = 3.e7
    as = 11.72
    ar = 841.99667
    bi = 1.
    bc = 2.
    bs = 0.41
    br = 0.8
    if (ihail==0) then
      ag = 19.3
      bg = 0.37
    else ! (matsun and huggins 1980)
      ag = 114.5 
      bg = 0.5
    endif
    ! constants and parameters
    rhosu = 85000./(287.15*273.15)
    rhow = 997.
    rhoi = 500.
    rhosn = 100.
    if (ihail==0) then
      rhog = 400.
    else
      rhog = 900.
    endif
    aimm   = 0.66
    bimm   = 100.
    ecr    = 1.
    dcs    = 125.e-6
    mi0    = 4./3.*pi*rhoi*pow(10.e-6,3)
    mg0    = 1.6e-10
    f1s    = 0.86
    f2s    = 0.28
    f1r    = 0.78
    f2r    = 0.308
    qsmall = 1.e-14
    eii    = 0.1
    eci    = 0.7
    cpw    = 4187.
    ! size distribution parameters
    ci = rhoi*pi/6.
    di = 3.
    cs = rhosn*pi/6.
    ds = 3.
    cg = rhog*pi/6.
    dg = 3.
    ! radius of contact nuclei
    rin = 0.1e-6
    mmult = 4./3.*pi*rhoi*pow(5.e-6,3)
    ! size limits for lambda
    lammaxi = 1./1.e-6
    lammini = 1./(2.*dcs+100.e-6)
    lammaxr = 1./20.e-6
    lamminr = 1./2800.e-6
    lammaxs = 1./10.e-6
    lammins = 1./2000.e-6
    lammaxg = 1./20.e-6
    lamming = 1./2000.e-6
    ! note: these parameters only used by the non-wrf-chem version of the 
    !       scheme with predicted droplet number
    ! ccn spectra for iact = 1
    ! maritime
    ! modified from rasmussen et al. 2002
    ! nccn = c*s^k, nccn is in cm-3, s is supersaturation ratio in %
    k1 = 0.4
    c1 = 120. 
    ! continental
    ! aerosol activation parameters for iact = 2
    ! parameters currently set for ammonium sulfate
    mw = 0.018
    osm = 1.
    vi = 3.
    epsm = 0.7
    rhoa = 1777.
    map = 0.132
    ma = 0.0284
    rr = 8.3145
    bact = vi*osm*epsm*mw*rhoa/(map*rhow)
    ! aerosol size distribution parameters currently set for mpace 
    ! (see morrison et al. 2007, jgr)
    ! mode 1
    rm1 = 0.052e-6
    sig1 = 2.04
    nanew1 = 72.2e6
    f11 = 0.5*exp(2.5*pow(log(sig1),2))
    f21 = 1.+0.25*log(sig1)
    ! mode 2
    rm2 = 1.3e-6
    sig2 = 2.5
    nanew2 = 1.8e6
    f12 = 0.5*exp(2.5*pow(log(sig2),2))
    f22 = 1.+0.25*log(sig2)
    ! constants for efficiency
    cons1  = gamma(1.+ds)*cs
    cons2  = gamma(1.+dg)*cg
    cons3  = gamma(4.+bs)/6.
    cons4  = gamma(4.+br)/6.
    cons5  = gamma(1.+bs)
    cons6  = gamma(1.+br)
    cons7  = gamma(4.+bg)/6.
    cons8  = gamma(1.+bg)
    cons9  = gamma(5./2.+br/2.)
    cons10 = gamma(5./2.+bs/2.)
    cons11 = gamma(5./2.+bg/2.)
    cons12 = gamma(1.+di)*ci
    cons13 = gamma(bs+3.)*pi/4.*eci
    cons14 = gamma(bg+3.)*pi/4.*eci
    cons15 = -1108.*eii*pow(pi,(1.-bs)/3.)*pow(rhosn,(-2.-bs)/3.)/(4.*720.)
    cons16 = gamma(bi+3.)*pi/4.*eci
    cons17 = 4.*2.*3.*rhosu*pi*eci*eci*gamma(2.*bs+2.)/(8.*(rhog-rhosn))
    cons18 = rhosn*rhosn
    cons19 = rhow*rhow
    cons20 = 20.*pi*pi*rhow*bimm
    cons21 = 4./(dcs*rhoi)
    cons22 = pi*rhoi*pow(dcs,3)/6.
    cons23 = pi/4.*eii*gamma(bs+3.)
    cons24 = pi/4.*ecr*gamma(br+3.)
    cons25 = pi*pi/24.*rhow*ecr*gamma(br+6.)
    cons26 = pi/6.*rhow
    cons27 = gamma(1.+bi)
    cons28 = gamma(4.+bi)/6.
    cons29 = 4./3.*pi*rhow*pow(25.e-6,3)
    cons30 = 4./3.*pi*rhow
    cons31 = pi*pi*ecr*rhosn
    cons32 = pi/2.*ecr
    cons33 = pi*pi*ecr*rhog
    cons34 = 5./2.+br/2.
    cons35 = 5./2.+bs/2.
    cons36 = 5./2.+bg/2.
    cons37 = 4.*pi*1.38e-23/(6.*pi*rin)
    cons38 = pi*pi/3.*rhow
    cons39 = pi*pi/36.*rhow*bimm
    cons40 = pi/6.*bimm
    cons41 = pi*pi*ecr*rhow
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
    call radar_init()
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
  ! reflectivity currently not included!!!!
  ! refl_10cm - calculated radar reflectivity at 10 cm (dbz)
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
  subroutine mp_morr_two_moment(itimestep,                                  &
                                th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, &
                                rho, pii, p, dt_in, dz, w,                  &
                                rainnc, rainncv, sr,                        &
                                snownc,snowncv,graupelnc,graupelncv,        &
                                refl_10cm, diagflag, do_radar_ref,          &
                                qrcuten, qscuten, qicuten,                  &
                                f_qndrop, qndrop,                           &
                                ids,ide,kds,kde,                            &
                                ims,ime,kms,kme,                            &
                                its,ite,kts,kte,                            &
                                wetscav_on, rainprod, evapprod,             &
                                qlsink,precr,preci,precs,precg ) bind(c,name="mp_morr_two_moment")

    use iso_c_binding, only: c_int, c_float
    implicit none
    integer(c_int  ), intent(in   ) :: ids, ide, kds, kde , &
                                       ims, ime, kms, kme , &
                                       its, ite, kts, kte
    real   (c_float), dimension(ims:ime, kms:kme), intent(inout) :: qv, qc, qr, qi, qs, qg, ni, ns, nr, th, ng   
    real   (c_float), dimension(ims:ime, kms:kme), intent(inout) :: qndrop
    real   (c_float), dimension(ims:ime, kms:kme), intent(inout) :: qlsink, rainprod, evapprod, preci, precs, precg, precr
    real   (c_float), dimension(ims:ime, kms:kme), intent(in   ) :: pii, p, dz, rho, w
    real   (c_float), intent(in) :: dt_in
    integer(c_int  ), intent(in):: itimestep
    real   (c_float), dimension(ims:ime), intent(inout) :: rainnc, rainncv, sr, snownc, snowncv, graupelnc, graupelncv
    real   (c_float), dimension(ims:ime, kms:kme), intent(inout) :: refl_10cm
    integer(c_int  ), intent(in) :: wetscav_on
    integer(c_int  ), intent(in) :: f_qndrop
    ! local variables
    real, dimension(its:ite, kts:kte) :: effi, effs, effr, effg
    real, dimension(its:ite, kts:kte) :: t,  effc
    real, dimension(kts:kte) :: qc_tend1d, qi_tend1d, qni_tend1d, qr_tend1d, &
                                ni_tend1d, ns_tend1d, nr_tend1d,             &
                                qc1d, qi1d, qr1d,ni1d, ns1d, nr1d, qs1d,     &
                                t_tend1d,qv_tend1d, t1d, qv1d, p1d, w1d,     &
                                effc1d, effi1d, effs1d, effr1d,dz1d,         &
                                qg_tend1d, ng_tend1d, qg1d, ng1d, effg1d,    &
                                qgsten,qrsten, qisten, qnisten, qcsten,      &
                                qrcu1d, qscu1d, qicu1d
    real, dimension(ims:ime, kms:kme), intent(in):: qrcuten, qscuten, qicuten
    logical :: flag_qndrop
    integer :: iinum
    real, dimension(kts:kte) :: nc1d, nc_tend1d,c2prec,csed,ised,ssed,gsed,rsed    
    real, dimension(kts:kte) :: rainprod1d, evapprod1d
    real, dimension(kts:kte) :: dbz
    real precprt1d, snowrt1d, snowprt1d, grplprt1d ! hm added 7/13/13
    integer i,k
    real dt
    integer(c_int), intent(in) :: diagflag
    integer(c_int), intent(in) :: do_radar_ref
    logical :: has_wetscav
    flag_qndrop = .false.
    flag_qndrop = (f_qndrop == 1)
    has_wetscav = (wetscav_on == 1)
    dt = dt_in   
    do i=its,ite
    do k=kts,kte
        t(i,k)        = th(i,k)*pii(i,k)
    enddo
    enddo

    do i=its,ite
      do k=kts,kte
        qc_tend1d (k) = 0.
        qi_tend1d (k) = 0.
        qni_tend1d(k) = 0.
        qr_tend1d (k) = 0.
        ni_tend1d (k) = 0.
        ns_tend1d (k) = 0.
        nr_tend1d (k) = 0.
        t_tend1d  (k) = 0.
        qv_tend1d (k) = 0.
        nc_tend1d (k) = 0. 
        qg_tend1d (k) = 0.
        ng_tend1d (k) = 0.
        nc1d      (k) = 0.
        qc1d      (k) = qc     (i,k)
        qi1d      (k) = qi     (i,k)
        qs1d      (k) = qs     (i,k)
        qr1d      (k) = qr     (i,k)
        ni1d      (k) = ni     (i,k)
        ns1d      (k) = ns     (i,k)
        nr1d      (k) = nr     (i,k)
        qg1d      (k) = qg     (i,k)
        ng1d      (k) = ng     (i,k)
        t1d       (k) = t      (i,k)
        qv1d      (k) = qv     (i,k)
        p1d       (k) = p      (i,k)
        dz1d      (k) = dz     (i,k)
        w1d       (k) = w      (i,k)
        qrcu1d    (k) = qrcuten(i,k)
        qscu1d    (k) = qscuten(i,k)
        qicu1d    (k) = qicuten(i,k)
      enddo
      iinum = 1
      call morr_two_moment_micro(qc_tend1d, qi_tend1d, qni_tend1d, qr_tend1d,   &
                                 ni_tend1d, ns_tend1d, nr_tend1d,               &
                                 qc1d, qi1d, qs1d, qr1d,ni1d, ns1d, nr1d,       &
                                 t_tend1d,qv_tend1d, t1d, qv1d, p1d, dz1d, w1d, &
                                 precprt1d,snowrt1d,                            &
                                 snowprt1d,grplprt1d,                           &
                                 effc1d,effi1d,effs1d,effr1d,dt,                &
                                 ims,ime,1,1,kms,kme,                           &
                                 its,ite,1,1,kts,kte,                           &
                                 qg_tend1d,ng_tend1d,qg1d,ng1d,effg1d,          &
                                 qrcu1d, qscu1d, qicu1d,                        &
                                 qgsten,qrsten,qisten,qnisten,qcsten,           &
                                 nc1d, nc_tend1d, iinum, c2prec,csed,ised,ssed,gsed,rsed )
      do k=kts,kte
        qc  (i,k) = qc1d  (k)
        qi  (i,k) = qi1d  (k)
        qs  (i,k) = qs1d  (k)
        qr  (i,k) = qr1d  (k)
        ni  (i,k) = ni1d  (k)
        ns  (i,k) = ns1d  (k)          
        nr  (i,k) = nr1d  (k)
        qg  (i,k) = qg1d  (k)
        ng  (i,k) = ng1d  (k)
        t   (i,k) = t1d   (k)
        qv  (i,k) = qv1d  (k)
        effc(i,k) = effc1d(k)
        effi(i,k) = effi1d(k)
        effs(i,k) = effs1d(k)
        effr(i,k) = effr1d(k)
        effg(i,k) = effg1d(k)
        th  (i,k) = t(i,k)/pii(i,k)
        if(qc(i,k)>1.e-10) then
           qlsink(i,k) = c2prec(k)/qc(i,k)
        else
           qlsink(i,k) = 0.0
        endif
        precr(i,k) = rsed(k)
        preci(i,k) = ised(k)
        precs(i,k) = ssed(k)
        precg(i,k) = gsed(k)
      enddo
      rainnc    (i) = rainnc(i)+precprt1d
      rainncv   (i) = precprt1d
      snownc    (i) = snownc(i)+snowprt1d
      snowncv   (i) = snowprt1d
      graupelnc (i) = graupelnc(i)+grplprt1d
      graupelncv(i) = grplprt1d
      sr        (i) = snowrt1d/(precprt1d+1.e-12)
    enddo   

  end subroutine mp_morr_two_moment



  ! this program is the main two-moment microphysics subroutine described by
  ! morrison et al. 2005 jas and morrison et al. 2009 mwr
  ! this scheme is a bulk double-moment scheme that predicts mixing
  ! ratios and number concentrations of five hydrometeor species:
  ! cloud droplets, cloud (small) ice, rain, snow, and graupel/hail.
  ! code structure: main subroutine is 'morr_two_moment'. also included in this file is
  ! 'function polysvp', 'function derf1', and
  ! 'function gamma'.
  ! note: this subroutine uses 1d array in vertical (column), even though variables are called '3d'......
  subroutine morr_two_moment_micro(qc3dten,qi3dten,qni3dten,qr3dten,                             &
                                   ni3dten,ns3dten,nr3dten,qc3d,qi3d,qni3d,qr3d,ni3d,ns3d,nr3d,  &
                                   t3dten,qv3dten,t3d,qv3d,pres,dzq,w3d,precrt,snowrt,           &
                                   snowprt,grplprt,                                              &
                                   effc,effi,effs,effr,dt,                                       &
                                   ims,ime, jms,jme, kms,kme,                                    &
                                   its,ite, jts,jte, kts,kte,                                    &
                                   qg3dten,ng3dten,qg3d,ng3d,effg,qrcu1d,qscu1d, qicu1d,         &
                                   qgsten,qrsten,qisten,qnisten,qcsten,                          &
                                   nc3d,nc3dten,iinum,                                           &
                                   c2prec,csed,ised,ssed,gsed,rsed )
    implicit none
    integer, intent(in   ) :: ims,ime, jms,jme, kms,kme , its,ite, jts,jte, kts,kte
    integer, intent(in   ) :: iinum
    real   , intent(in   ) :: dt                            ! model time step (sec)
    real   , intent(inout) :: precrt                        ! total precip per time step (mm)
    real   , intent(inout) :: snowrt                        ! snow per time step (mm)
    real   , intent(inout) :: snowprt                       ! total cloud ice plus snow per time step (mm)
    real   , intent(inout) :: grplprt                       ! total graupel per time step (mm)
    real   , intent(inout), dimension(kts:kte) :: qc3dten   ! cloud water mixing ratio tendency (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qi3dten   ! cloud ice mixing ratio tendency (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qni3dten  ! snow mixing ratio tendency (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qr3dten   ! rain mixing ratio tendency (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: ni3dten   ! cloud ice number concentration (1/kg/s)
    real   , intent(inout), dimension(kts:kte) :: ns3dten   ! snow number concentration (1/kg/s)
    real   , intent(inout), dimension(kts:kte) :: nr3dten   ! rain number concentration (1/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qc3d      ! cloud water mixing ratio (kg/kg)
    real   , intent(inout), dimension(kts:kte) :: qi3d      ! cloud ice mixing ratio (kg/kg)
    real   , intent(inout), dimension(kts:kte) :: qni3d     ! snow mixing ratio (kg/kg)
    real   , intent(inout), dimension(kts:kte) :: qr3d      ! rain mixing ratio (kg/kg)
    real   , intent(inout), dimension(kts:kte) :: ni3d      ! cloud ice number concentration (1/kg)
    real   , intent(inout), dimension(kts:kte) :: ns3d      ! snow number concentration (1/kg)
    real   , intent(inout), dimension(kts:kte) :: nr3d      ! rain number concentration (1/kg)
    real   , intent(inout), dimension(kts:kte) :: t3dten    ! temperature tendency (k/s)
    real   , intent(inout), dimension(kts:kte) :: qv3dten   ! water vapor mixing ratio tendency (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: t3d       ! temperature (k)
    real   , intent(inout), dimension(kts:kte) :: qv3d      ! water vapor mixing ratio (kg/kg)
    real   , intent(inout), dimension(kts:kte) :: pres      ! atmospheric pressure (pa)
    real   , intent(inout), dimension(kts:kte) :: dzq       ! difference in height across level (m)
    real   , intent(inout), dimension(kts:kte) :: w3d       ! grid-scale vertical velocity (m/s)
    real   , intent(inout), dimension(kts:kte) :: nc3d
    real   , intent(inout), dimension(kts:kte) :: nc3dten
    real   , intent(inout), dimension(kts:kte) :: qg3dten   ! graupel mix ratio tendency (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: ng3dten   ! graupel numb conc tendency (1/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qg3d      ! graupel mix ratio (kg/kg)
    real   , intent(inout), dimension(kts:kte) :: ng3d      ! graupel number conc (1/kg)
    real   , intent(inout), dimension(kts:kte) :: qgsten    ! graupel sed tend (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qrsten    ! rain sed tend (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qisten    ! cloud ice sed tend (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qnisten   ! snow sed tend (kg/kg/s)
    real   , intent(inout), dimension(kts:kte) :: qcsten    ! cloud wat sed tend (kg/kg/s)      
    real   , intent(inout), dimension(kts:kte) :: qrcu1d
    real   , intent(inout), dimension(kts:kte) :: qscu1d
    real   , intent(inout), dimension(kts:kte) :: qicu1d
    real   , intent(inout), dimension(kts:kte) :: effc      ! droplet effective radius (micron)
    real   , intent(inout), dimension(kts:kte) :: effi      ! cloud ice effective radius (micron)
    real   , intent(inout), dimension(kts:kte) :: effs      ! snow effective radius (micron)
    real   , intent(inout), dimension(kts:kte) :: effr      ! rain effective radius (micron)
    real   , intent(inout), dimension(kts:kte) :: effg      ! graupel effective radius (micron)
    real, dimension(kts:kte) :: lamc      ! slope parameter for droplets (m-1)
    real, dimension(kts:kte) :: lami      ! slope parameter for cloud ice (m-1)
    real, dimension(kts:kte) :: lams      ! slope parameter for snow (m-1)
    real, dimension(kts:kte) :: lamr      ! slope parameter for rain (m-1)
    real, dimension(kts:kte) :: lamg      ! slope parameter for graupel (m-1)
    real, dimension(kts:kte) :: cdist1    ! psd parameter for droplets
    real, dimension(kts:kte) :: n0i       ! intercept parameter for cloud ice (kg-1 m-1)
    real, dimension(kts:kte) :: n0s       ! intercept parameter for snow (kg-1 m-1)
    real, dimension(kts:kte) :: n0rr      ! intercept parameter for rain (kg-1 m-1)
    real, dimension(kts:kte) :: n0g       ! intercept parameter for graupel (kg-1 m-1)
    real, dimension(kts:kte) :: pgam      ! spectral shape parameter for droplets
    real, dimension(kts:kte) :: nsubc     ! loss of nc during evap
    real, dimension(kts:kte) :: nsubi     ! loss of ni during sub.
    real, dimension(kts:kte) :: nsubs     ! loss of ns during sub.
    real, dimension(kts:kte) :: nsubr     ! loss of nr during evap
    real, dimension(kts:kte) :: prd       ! dep cloud ice
    real, dimension(kts:kte) :: pre       ! evap of rain
    real, dimension(kts:kte) :: prds      ! dep snow
    real, dimension(kts:kte) :: nnuccc    ! change n due to contact freez droplets
    real, dimension(kts:kte) :: mnuccc    ! change q due to contact freez droplets
    real, dimension(kts:kte) :: pra       ! accretion droplets by rain
    real, dimension(kts:kte) :: prc       ! autoconversion droplets
    real, dimension(kts:kte) :: pcc       ! cond/evap droplets
    real, dimension(kts:kte) :: nnuccd    ! change n freezing aerosol (prim ice nucleation)
    real, dimension(kts:kte) :: mnuccd    ! change q freezing aerosol (prim ice nucleation)
    real, dimension(kts:kte) :: mnuccr    ! change q due to contact freez rain
    real, dimension(kts:kte) :: nnuccr    ! change n due to contact freez rain
    real, dimension(kts:kte) :: npra      ! change in n due to droplet acc by rain
    real, dimension(kts:kte) :: nragg     ! self-collection/breakup of rain
    real, dimension(kts:kte) :: nsagg     ! self-collection of snow
    real, dimension(kts:kte) :: nprc      ! change nc autoconversion droplets
    real, dimension(kts:kte) :: nprc1     ! change nr autoconversion droplets
    real, dimension(kts:kte) :: prai      ! change q accretion cloud ice by snow
    real, dimension(kts:kte) :: prci      ! change q autoconversin cloud ice to snow
    real, dimension(kts:kte) :: psacws    ! change q droplet accretion by snow
    real, dimension(kts:kte) :: npsacws   ! change n droplet accretion by snow
    real, dimension(kts:kte) :: psacwi    ! change q droplet accretion by cloud ice
    real, dimension(kts:kte) :: npsacwi   ! change n droplet accretion by cloud ice
    real, dimension(kts:kte) :: nprci     ! change n autoconversion cloud ice by snow
    real, dimension(kts:kte) :: nprai     ! change n accretion cloud ice
    real, dimension(kts:kte) :: nmults    ! ice mult due to riming droplets by snow
    real, dimension(kts:kte) :: nmultr    ! ice mult due to riming rain by snow
    real, dimension(kts:kte) :: qmults    ! change q due to ice mult droplets/snow
    real, dimension(kts:kte) :: qmultr    ! change q due to ice rain/snow
    real, dimension(kts:kte) :: pracs     ! change q rain-snow collection
    real, dimension(kts:kte) :: npracs    ! change n rain-snow collection
    real, dimension(kts:kte) :: pccn      ! change q droplet activation
    real, dimension(kts:kte) :: psmlt     ! change q melting snow to rain
    real, dimension(kts:kte) :: evpms     ! chnage q melting snow evaporating
    real, dimension(kts:kte) :: nsmlts    ! change n melting snow
    real, dimension(kts:kte) :: nsmltr    ! change n melting snow to rain
    real, dimension(kts:kte) :: piacr     ! change qr, ice-rain collection
    real, dimension(kts:kte) :: niacr     ! change n, ice-rain collection
    real, dimension(kts:kte) :: praci     ! change qi, ice-rain collection
    real, dimension(kts:kte) :: piacrs    ! change qr, ice rain collision, added to snow
    real, dimension(kts:kte) :: niacrs    ! change n, ice rain collision, added to snow
    real, dimension(kts:kte) :: pracis    ! change qi, ice rain collision, added to snow
    real, dimension(kts:kte) :: eprd      ! sublimation cloud ice
    real, dimension(kts:kte) :: eprds     ! sublimation snow
    real, dimension(kts:kte) :: pracg     ! change in q collection rain by graupel
    real, dimension(kts:kte) :: psacwg    ! change in q collection droplets by graupel
    real, dimension(kts:kte) :: pgsacw    ! conversion q to graupel due to collection droplets by snow
    real, dimension(kts:kte) :: pgracs    ! conversion q to graupel due to collection rain by snow
    real, dimension(kts:kte) :: prdg      ! dep of graupel
    real, dimension(kts:kte) :: eprdg     ! sub of graupel
    real, dimension(kts:kte) :: evpmg     ! change q melting of graupel and evaporation
    real, dimension(kts:kte) :: pgmlt     ! change q melting of graupel
    real, dimension(kts:kte) :: npracg    ! change n collection rain by graupel
    real, dimension(kts:kte) :: npsacwg   ! change n collection droplets by graupel
    real, dimension(kts:kte) :: nscng     ! change n conversion to graupel due to collection droplets by snow
    real, dimension(kts:kte) :: ngracs    ! change n conversion to graupel due to collection rain by snow
    real, dimension(kts:kte) :: ngmltg    ! change n melting graupel
    real, dimension(kts:kte) :: ngmltr    ! change n melting graupel to rain
    real, dimension(kts:kte) :: nsubg     ! change n sub/dep of graupel
    real, dimension(kts:kte) :: psacr     ! conversion due to coll of snow by rain
    real, dimension(kts:kte) :: nmultg    ! ice mult due to acc droplets by graupel
    real, dimension(kts:kte) :: nmultrg   ! ice mult due to acc rain by graupel
    real, dimension(kts:kte) :: qmultg    ! change q due to ice mult droplets/graupel
    real, dimension(kts:kte) :: qmultrg   ! change q due to ice mult rain/graupel
    real, dimension(kts:kte) :: kap       ! thermal conductivity of air
    real, dimension(kts:kte) :: evs       ! saturation vapor pressure
    real, dimension(kts:kte) :: eis       ! ice saturation vapor pressure
    real, dimension(kts:kte) :: qvs       ! saturation mixing ratio
    real, dimension(kts:kte) :: qvi       ! ice saturation mixing ratio
    real, dimension(kts:kte) :: qvqvs     ! sautration ratio
    real, dimension(kts:kte) :: qvqvsi    ! ice saturaion ratio
    real, dimension(kts:kte) :: dv        ! diffusivity of water vapor in air
    real, dimension(kts:kte) :: xxls      ! latent heat of sublimation
    real, dimension(kts:kte) :: xxlv      ! latent heat of vaporization
    real, dimension(kts:kte) :: cpm       ! specific heat at const pressure for moist air
    real, dimension(kts:kte) :: mu        ! viscocity of air
    real, dimension(kts:kte) :: sc        ! schmidt number
    real, dimension(kts:kte) :: xlf       ! latent heat of freezing
    real, dimension(kts:kte) :: rho       ! air density
    real, dimension(kts:kte) :: ab        ! correction to condensation rate due to latent heating
    real, dimension(kts:kte) :: abi       ! correction to deposition rate due to latent heating
    real, dimension(kts:kte) :: dap       ! diffusivity of aerosol
    real                     :: nacnt     ! number of contact in
    real                     :: fmult     ! temp.-dep. parameter for rime-splintering
    real                     :: coffi     ! ice autoconversion parameter
    real                     :: uni, umi,umr
    real                     :: rgvm
    real                     :: faltndr,faltndi,faltndni,rho2
    real                     :: ums,uns
    real                     :: faltnds,faltndns,unr,faltndg,faltndng
    real                     :: unc,umc,ung,umg
    real                     :: faltndc,faltndnc
    real                     :: faltndnr
    real, dimension(kts:kte) :: dumi,dumr,dumfni,dumg,dumfng
    real, dimension(kts:kte) :: fr, fi, fni,fg,fng
    real, dimension(kts:kte) :: faloutr,falouti,faloutni
    real, dimension(kts:kte) :: dumqs,dumfns
    real, dimension(kts:kte) :: fs,fns, falouts,faloutns,faloutg,faloutng
    real, dimension(kts:kte) :: dumc,dumfnc
    real, dimension(kts:kte) :: fc,faloutc,faloutnc
    real, dimension(kts:kte) :: fnc,dumfnr,faloutnr
    real, dimension(kts:kte) :: fnr
    real, dimension(kts:kte) ::    ain,arn,asn,acn,agn
    real                     :: dum,dum1,dum2,dumt,dumqv,dumqss,dumqsi,dums
    real                     :: dqsdt    ! change of sat. mix. rat. with temperature
    real                     :: dqsidt   ! change in ice sat. mixing rat. with t
    real                     :: epsi     ! 1/phase rel. time (see m2005), ice
    real                     :: epss     ! 1/phase rel. time (see m2005), snow
    real                     :: epsr     ! 1/phase rel. time (see m2005), rain
    real                     :: epsg     ! 1/phase rel. time (see m2005), graupel
    real                     :: tauc     ! phase rel. time (see m2005), droplets
    real                     :: taur     ! phase rel. time (see m2005), rain
    real                     :: taui     ! phase rel. time (see m2005), cloud ice
    real                     :: taus     ! phase rel. time (see m2005), snow
    real                     :: taug     ! phase rel. time (see m2005), graupel
    real                     :: dumact, dum3
    integer                  :: k,nstep,n
    integer                  :: ltrue
    real                     :: ct        ! droplet activation parameter
    real                     :: temp1     ! dummy temperature
    real                     :: sat1      ! dummy saturation
    real                     :: sigvl     ! surface tension liq/vapor
    real                     :: kel       ! kelvin parameter
    real                     :: kc2       ! total ice nucleation rate
    real                     :: cry,kry   ! aerosol activation parameters
    real                     :: dumqi,dumni,dc0,ds0,dg0
    real                     :: dumqc,dumqr,ratio,sum_dep,fudgef
    real                     :: wef
    real                     :: anuc,bnuc
    real                     :: aact,gamm,gg,psi,eta1,eta2,sm1,sm2,smax,uu1,uu2,alpha
    real                     :: dlams,dlamr,dlami,dlamc,dlamg,lammax,lammin
    integer                  :: idrop
    real, dimension(kts:kte) :: c2prec,csed,ised,ssed,gsed,rsed
    real, dimension(kts:kte) :: tqimelt ! melting of cloud ice (tendency)
    ltrue = 0
    do k = kts,kte
      nc3dten(k) = 0.
      c2prec (k) = 0.
      csed   (k) = 0.
      ised   (k) = 0.
      ssed   (k) = 0.
      gsed   (k) = 0.
      rsed   (k) = 0.
      xxlv   (k) = 3.1484e6-2370.*t3d(k)
      xxls   (k) = 3.15e6-2370.*t3d(k)+0.3337e6
      cpm    (k) = cp*(1.+0.887*qv3d(k))
      evs    (k) = min(0.99*pres(k),polysvp(t3d(k),0))   ! pa
      eis    (k) = min(0.99*pres(k),polysvp(t3d(k),1))   ! pa
      if (eis(k) > evs(k)) eis(k) = evs(k)
      qvs    (k) = ep_2*evs(k)/(pres(k)-evs(k))
      qvi    (k) = ep_2*eis(k)/(pres(k)-eis(k))
      qvqvs  (k) = qv3d(k)/qvs(k)
      qvqvsi (k) = qv3d(k)/qvi(k)
      rho    (k) = pres(k)/(r*t3d(k))
      if (qrcu1d(k) >= 1.e-10) then
        dum = 1.8e5*pow(qrcu1d(k)*dt/(pi*rhow*pow(rho(k),3)),0.25)
        nr3d(k) = nr3d(k)+dum
      endif
      if (qscu1d(k) >= 1.e-10) then
        dum = 3.e5*pow(qscu1d(k)*dt/(cons1*pow(rho(k),3)),1./(ds+1.))
        ns3d(k) = ns3d(k)+dum
      endif
      if (qicu1d(k) >= 1.e-10) then
        dum = qicu1d(k)*dt/(ci*pow(80.e-6,di))
        ni3d(k) = ni3d(k)+dum
      endif
      if (qvqvs(k) < 0.9) then
        if (qr3d(k) < 1.e-8) then
           qv3d(k)=qv3d(k)+qr3d(k)
           t3d (k)=t3d(k)-qr3d(k)*xxlv(k)/cpm(k)
           qr3d(k)=0.
        endif
        if (qc3d(k) < 1.e-8) then
           qv3d(k)=qv3d(k)+qc3d(k)
           t3d (k)=t3d(k)-qc3d(k)*xxlv(k)/cpm(k)
           qc3d(k)=0.
        endif
      endif

      if (qvqvsi(k) < 0.9) then
        if (qi3d(k) < 1.e-8) then
           qv3d(k)=qv3d(k)+qi3d(k)
           t3d (k)=t3d(k)-qi3d(k)*xxls(k)/cpm(k)
           qi3d(k)=0.
        endif
        if (qni3d(k) < 1.e-8) then
           qv3d (k)=qv3d(k)+qni3d(k)
           t3d  (k)=t3d(k)-qni3d(k)*xxls(k)/cpm(k)
           qni3d(k)=0.
        endif
        if (qg3d(k) < 1.e-8) then
           qv3d(k)=qv3d(k)+qg3d(k)
           t3d (k)=t3d(k)-qg3d(k)*xxls(k)/cpm(k)
           qg3d(k)=0.
        endif
      endif
      xlf(k) = xxls(k)-xxlv(k)
      if (qc3d(k) < qsmall) then
        qc3d(k) = 0.
        nc3d(k) = 0.
        effc(k) = 0.
      endif
      if (qr3d(k) < qsmall) then
        qr3d(k) = 0.
        nr3d(k) = 0.
        effr(k) = 0.
      endif
      if (qi3d(k) < qsmall) then
        qi3d(k) = 0.
        ni3d(k) = 0.
        effi(k) = 0.
      endif
      if (qni3d(k) < qsmall) then
        qni3d(k) = 0.
        ns3d(k) = 0.
        effs(k) = 0.
      endif
      if (qg3d(k) < qsmall) then
        qg3d(k) = 0.
        ng3d(k) = 0.
        effg(k) = 0.
      endif
      qrsten (k) = 0.
      qisten (k) = 0.
      qnisten(k) = 0.
      qcsten (k) = 0.
      qgsten (k) = 0.
      mu     (k) = 1.496e-6*pow(t3d(k),1.5)/(t3d(k)+120.)
      dum        = pow(rhosu/rho(k),0.54)
      ain    (k) = pow(rhosu/rho(k),0.35)*ai
      arn    (k) = dum*ar
      asn    (k) = dum*as
      acn    (k) = g*rhow/(18.*mu(k))
      agn    (k) = dum*ag
      lami   (k) = 0.
      if (qc3d(k) < qsmall.and.qi3d(k) < qsmall.and.qni3d(k) < qsmall .and.qr3d(k) < qsmall.and.qg3d(k) < qsmall) then
        if (t3d(k) <  273.15.and.qvqvsi(k) < 0.999) cycle
        if (t3d(k) >= 273.15.and.qvqvs (k) < 0.999) cycle
      endif
      kap   (k) = 1.414e3*mu(k)
      dv    (k) = 8.794e-5*pow(t3d(k),1.81)/pres(k)
      sc    (k) = mu(k)/(rho(k)*dv(k))
      dum       = (rv*pow(t3d(k),2))
      dqsdt     = xxlv(k)*qvs(k)/dum
      dqsidt    = xxls(k)*qvi(k)/dum
      abi   (k) = 1.+dqsidt*xxls(k)/cpm(k)
      ab    (k) = 1.+dqsdt*xxlv(k)/cpm(k)
      if (t3d(k) >= 273.15) then
        if (iinum==1) then
          nc3d(k) = ndcnst*1.e6/rho(k)
        endif
        if (qni3d(k) < 1.e-6) then
          qr3d (k) = qr3d(k)+qni3d(k)
          nr3d (k) = nr3d(k)+ns3d (k)
          t3d  (k) = t3d (k)-qni3d(k)*xlf(k)/cpm(k)
          qni3d(k) = 0.
          ns3d (k) = 0.
        endif
        if (qg3d(k) < 1.e-6) then
          qr3d(k) = qr3d(k)+qg3d(k)
          nr3d(k) = nr3d(k)+ng3d(k)
          t3d (k) = t3d (k)-qg3d(k)*xlf(k)/cpm(k)
          qg3d(k) = 0.
          ng3d(k) = 0.
        endif
        if (.not. (qc3d(k) < qsmall.and.qni3d(k) < 1.e-8.and.qr3d(k) < qsmall.and.qg3d(k) < 1.e-8)) then
          ns3d(k) = max(0.,ns3d(k))
          nc3d(k) = max(0.,nc3d(k))
          nr3d(k) = max(0.,nr3d(k))
          ng3d(k) = max(0.,ng3d(k))
          if (qr3d(k) >= qsmall) then
            lamr(k) = pow(pi*rhow*nr3d(k)/qr3d(k),1./3.)
            n0rr(k) = nr3d(k)*lamr(k)
            if (lamr(k) < lamminr) then
              lamr(k) = lamminr
              n0rr(k) = pow(lamr(k),4)*qr3d(k)/(pi*rhow)
              nr3d(k) = n0rr(k)/lamr(k)
            else if (lamr(k) > lammaxr) then
              lamr(k) = lammaxr
              n0rr(k) = pow(lamr(k),4)*qr3d(k)/(pi*rhow)
              nr3d(k) = n0rr(k)/lamr(k)
            endif
          endif
          if (qc3d(k) >= qsmall) then
            dum     =  pres(k)/(287.15*t3d(k))
            pgam(k) = 0.0005714*(nc3d(k)/1.e6*dum)+0.2714
            pgam(k) = 1./(pow(pgam(k),2))-1.
            pgam(k) = max(pgam(k),2.)
            pgam(k) = min(pgam(k),10.)
            lamc(k) = pow(cons26*nc3d(k)*gamma(pgam(k)+4.)/(qc3d(k)*gamma(pgam(k)+1.)),1./3.)
            lammin  = (pgam(k)+1.)/60.e-6
            lammax  = (pgam(k)+1.)/1.e-6
            if (lamc(k) < lammin) then
              lamc(k) = lammin
              nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26
            else if (lamc(k) > lammax) then
              lamc(k) = lammax
              nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26
            endif
          endif
          if (qni3d(k) >= qsmall) then
            lams(k) = pow(cons1*ns3d(k)/qni3d(k),1./ds)
            n0s (k) = ns3d(k)*lams(k)
            if (lams(k) < lammins) then
              lams(k) = lammins
              n0s (k) = pow(lams(k),4)*qni3d(k)/cons1
              ns3d(k) = n0s(k)/lams(k)
            else if (lams(k) > lammaxs) then
              lams(k) = lammaxs
              n0s (k) = pow(lams(k),4)*qni3d(k)/cons1
              ns3d(k) = n0s(k)/lams(k)
            endif
          endif
          if (qg3d(k) >= qsmall) then
            lamg(k) = pow(cons2*ng3d(k)/qg3d(k),1./dg)
            n0g (k) = ng3d(k)*lamg(k)
            if (lamg(k) < lamming) then
              lamg(k) = lamming
              n0g (k) = pow(lamg(k),4)*qg3d(k)/cons2
              ng3d(k) = n0g(k)/lamg(k)
            else if (lamg(k) > lammaxg) then
              lamg(k) = lammaxg
              n0g (k) = pow(lamg(k),4)*qg3d(k)/cons2
              ng3d(k) = n0g(k)/lamg(k)
            endif
          endif
          prc   (k) = 0.
          nprc  (k) = 0.
          nprc1 (k) = 0.
          pra   (k) = 0.
          npra  (k) = 0.
          nragg (k) = 0.
          nsmlts(k) = 0.
          nsmltr(k) = 0.
          evpms (k) = 0.
          pcc   (k) = 0.
          pre   (k) = 0.
          nsubc (k) = 0.
          nsubr (k) = 0.
          pracg (k) = 0.
          npracg(k) = 0.
          psmlt (k) = 0.
          pgmlt (k) = 0.
          evpmg (k) = 0.
          pracs (k) = 0.
          npracs(k) = 0.
          ngmltg(k) = 0.
          ngmltr(k) = 0.
          if (qc3d(k) >= 1.e-6) then
            prc  (k)=1350.*pow(qc3d(k),2.47)*pow(nc3d(k)/1.e6*rho(k),-1.79)
            nprc1(k) = prc(k)/cons29
            nprc (k) = prc(k)/(qc3d(k)/nc3d(k))
            nprc (k) = min( nprc (k) , nc3d(k)/dt )
            nprc1(k) = min( nprc1(k) , nprc(k)    )
          endif
          if (qr3d(k) >= 1.e-8.and.qni3d(k) >= 1.e-8) then
            ums = asn(k)*cons3/pow(lams(k),bs)
            umr = arn(k)*cons4/pow(lamr(k),br)
            uns = asn(k)*cons5/pow(lams(k),bs)
            unr = arn(k)*cons6/pow(lamr(k),br)
            dum = pow(rhosu/rho(k),0.54)
            ums = min( ums , 1.2*dum )
            uns = min( uns , 1.2*dum )
            umr = min( umr , 9.1*dum )
            unr = min( unr , 9.1*dum )
            pracs(k) = cons41*(pow(pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(k)*n0rr(k)*n0s(k)/pow(lamr(k),3)* &
                        (5./(pow(lamr(k),3)*lams(k))+2./(pow(lamr(k),2)*pow(lams(k),2))+0.5/(lamr(k)*pow(lams(k),3))))
          endif
          if (qr3d(k) >= 1.e-8.and.qg3d(k) >= 1.e-8) then
            umg = agn(k)*cons7/pow(lamg(k),bg)
            umr = arn(k)*cons4/pow(lamr(k),br)
            ung = agn(k)*cons8/pow(lamg(k),bg)
            unr = arn(k)*cons6/pow(lamr(k),br)
            dum = pow(rhosu/rho(k),0.54)
            umg = min( umg , 20.*dum )
            ung = min( ung , 20.*dum )
            umr = min( umr , 9.1*dum )
            unr = min( unr , 9.1*dum )
            pracg(k)  = cons41*(pow(pow(1.2*umr-0.95*umg,2)+0.08*umg*umr,0.5)*rho(k)*n0rr(k)*n0g(k)/pow(lamr(k),3)* &
                        (5./(pow(lamr(k),3)*lamg(k))+2./(pow(lamr(k),2)*pow(lamg(k),2))+0.5/(lamr(k)*pow(lamg(k),3))))
            dum       = pracg(k)/5.2e-7
            npracg(k) = cons32*rho(k)*pow(1.7*pow(unr-ung,2)+0.3*unr*ung,0.5)*n0rr(k)*n0g(k)* &
                        (1./(pow(lamr(k),3)*lamg(k))+1./(pow(lamr(k),2)*pow(lamg(k),2))+1./(lamr(k)*pow(lamg(k),3)))
            npracg(k) = npracg(k)-dum
          endif
          if (qr3d(k) >= 1.e-8 .and. qc3d(k) >= 1.e-8) then
            dum     = (qc3d(k)*qr3d(k))
            pra (k) = 67.*pow(dum,1.15)
            npra(k) = pra(k)/(qc3d(k)/nc3d(k))
          endif
          if (qr3d(k) >= 1.e-8) then
            dum1=300.e-6
            if (1./lamr(k) < dum1) then
              dum=1.
            else if (1./lamr(k) >= dum1) then
              dum=2.-exp(2300.*(1./lamr(k)-dum1))
            endif
            nragg(k) = -5.78*dum*nr3d(k)*qr3d(k)*rho(k)
          endif
          if (qr3d(k) >= qsmall) then
            epsr = 2.*pi*n0rr(k)*rho(k)*dv(k)*(f1r/(lamr(k)*lamr(k))+f2r*pow(arn(k)*rho(k)/mu(k),0.5)*pow(sc(k),1./3.)*cons9/(pow(lamr(k),cons34)))
          else
            epsr = 0.
          endif
          if (qv3d(k) < qvs(k)) then
            pre(k) = epsr*(qv3d(k)-qvs(k))/ab(k)
            pre(k) = min(pre(k),0.)
          else
            pre(k) = 0.
          endif
          if (qni3d(k) >= 1.e-8) then
            dum      = -cpw/xlf(k)*(t3d(k)-273.15)*pracs(k)
            psmlt(k) = 2.*pi*n0s(k)*kap(k)*(273.15-t3d(k))/xlf(k)*(f1s/(lams(k)*lams(k))+f2s*pow(asn(k)*rho(k)/mu(k),0.5)*pow(sc(k),1./3.)*cons10/(pow(lams(k),cons35)))+dum
            if (qvqvs(k) < 1.) then
              epss     = 2.*pi*n0s(k)*rho(k)*dv(k)*(f1s/(lams(k)*lams(k))+f2s*pow(asn(k)*rho(k)/mu(k),0.5)*pow(sc(k),1./3.)*cons10/(pow(lams(k),cons35)))
              evpms(k) = (qv3d(k)-qvs(k))*epss/ab(k)    
              evpms(k) = max(evpms(k),psmlt(k))
              psmlt(k) = psmlt(k)-evpms(k)
            endif
          endif
          if (qg3d(k) >= 1.e-8) then
            dum      = -cpw/xlf(k)*(t3d(k)-273.15)*pracg(k)
            pgmlt(k) = 2.*pi*n0g(k)*kap(k)*(273.15-t3d(k))/xlf(k)*(f1s/(lamg(k)*lamg(k))+f2s*pow(agn(k)*rho(k)/mu(k),0.5)*pow(sc(k),1./3.)*cons11/(pow(lamg(k),cons36)))+dum
            if (qvqvs(k) < 1.) then
              epsg     = 2.*pi*n0g(k)*rho(k)*dv(k)*(f1s/(lamg(k)*lamg(k))+f2s*pow(agn(k)*rho(k)/mu(k),0.5)*pow(sc(k),1./3.)*cons11/(pow(lamg(k),cons36)))
              evpmg(k) = (qv3d(k)-qvs(k))*epsg/ab(k)
              evpmg(k) = max(evpmg(k),pgmlt(k))
              pgmlt(k) = pgmlt(k)-evpmg(k)
            endif
          endif
          pracg(k) = 0.
          pracs(k) = 0.
          dum = (prc(k)+pra(k))*dt
          if (dum > qc3d(k).and.qc3d(k) >= qsmall) then
            ratio = qc3d(k)/dum
            prc(k) = prc(k)*ratio
            pra(k) = pra(k)*ratio
          endif
          dum = (-psmlt(k)-evpms(k)+pracs(k))*dt
          if (dum > qni3d(k).and.qni3d(k) >= qsmall) then
            ratio    = qni3d(k)/dum
            psmlt(k) = psmlt(k)*ratio
            evpms(k) = evpms(k)*ratio
            pracs(k) = pracs(k)*ratio
          endif
          dum = (-pgmlt(k)-evpmg(k)+pracg(k))*dt
          if (dum > qg3d(k).and.qg3d(k) >= qsmall) then
            ratio    = qg3d (k)/dum
            pgmlt(k) = pgmlt(k)*ratio
            evpmg(k) = evpmg(k)*ratio
            pracg(k) = pracg(k)*ratio
          endif
          dum = (-pracs(k)-pracg(k)-pre(k)-pra(k)-prc(k)+psmlt(k)+pgmlt(k))*dt
          if (dum > qr3d(k).and.qr3d(k) >= qsmall) then
            ratio  = (qr3d(k)/dt+pracs(k)+pracg(k)+pra(k)+prc(k)-psmlt(k)-pgmlt(k))/(-pre(k))
            pre(k) = pre(k)*ratio
          endif
          qv3dten (k) = qv3dten (k) + (-pre(k)-evpms(k)-evpmg(k))
          t3dten  (k) = t3dten  (k) + (pre(k)*xxlv(k)+(evpms(k)+evpmg(k))*xxls(k)+(psmlt(k)+pgmlt(k)-pracs(k)-pracg(k))*xlf(k))/cpm(k)
          qc3dten (k) = qc3dten (k) + (-pra(k)-prc(k))
          qr3dten (k) = qr3dten (k) + (pre(k)+pra(k)+prc(k)-psmlt(k)-pgmlt(k)+pracs(k)+pracg(k))
          qni3dten(k) = qni3dten(k) + (psmlt(k)+evpms(k)-pracs(k))
          qg3dten (k) = qg3dten (k) + (pgmlt(k)+evpmg(k)-pracg(k))
          nc3dten (k) = nc3dten (k) + (-npra(k)-nprc(k))
          nr3dten (k) = nr3dten (k) + (nprc1(k)+nragg(k)-npracg(k))
          c2prec  (k) = pra(k)+prc(k)
          if (pre(k) < 0.) then
            dum      = pre(k)*dt/qr3d(k)
            dum      = max(-1.,dum)
            nsubr(k) = dum*nr3d(k)/dt
          endif
          if (evpms(k)+psmlt(k) < 0.) then
            dum       = (evpms(k)+psmlt(k))*dt/qni3d(k)
            dum       = max(-1.,dum)
            nsmlts(k) = dum*ns3d(k)/dt
          endif
          if (psmlt(k) < 0.) then
            dum       = psmlt(k)*dt/qni3d(k)
            dum       = max(-1.0,dum)
            nsmltr(k) = dum*ns3d(k)/dt
          endif
          if (evpmg(k)+pgmlt(k) < 0.) then
            dum       = (evpmg(k)+pgmlt(k))*dt/qg3d(k)
            dum       = max(-1.,dum)
            ngmltg(k) = dum*ng3d(k)/dt
          endif
          if (pgmlt(k) < 0.) then
            dum       = pgmlt(k)*dt/qg3d(k)
            dum       = max(-1.0,dum)
            ngmltr(k) = dum*ng3d(k)/dt
          endif
          ns3dten(k) = ns3dten(k)+(nsmlts(k))
          ng3dten(k) = ng3dten(k)+(ngmltg(k))
          nr3dten(k) = nr3dten(k)+(nsubr(k)-nsmltr(k)-ngmltr(k))
        endif

        dumt = t3d(k)+dt*t3dten(k)
        dumqv = qv3d(k)+dt*qv3dten(k)
        dum=min(0.99*pres(k),polysvp(dumt,0))
        dumqss = ep_2*dum/(pres(k)-dum)
        dumqc = qc3d(k)+dt*qc3dten(k)
        dumqc = max(dumqc,0.)
        dums = dumqv-dumqss
        pcc(k) = dums/(1.+pow(xxlv(k),2)*dumqss/(cpm(k)*rv*pow(dumt,2)))/dt
        if (pcc(k)*dt+dumqc < 0.) then
          pcc(k) = -dumqc/dt
        endif
        qv3dten(k) = qv3dten(k)-pcc(k)
        t3dten(k) = t3dten(k)+pcc(k)*xxlv(k)/cpm(k)
        qc3dten(k) = qc3dten(k)+pcc(k)
      else  ! temperature < 273.15
        if (iinum==1) then
          nc3d(k)=ndcnst*1.e6/rho(k)
        endif
        ni3d(k) = max(0.,ni3d(k))
        ns3d(k) = max(0.,ns3d(k))
        nc3d(k) = max(0.,nc3d(k))
        nr3d(k) = max(0.,nr3d(k))
        ng3d(k) = max(0.,ng3d(k))
        if (qi3d(k) >= qsmall) then
          lami(k) = pow(cons12*ni3d(k)/qi3d(k),1./di)
          n0i(k) = ni3d(k)*lami(k)
          if (lami(k) < lammini) then
            lami(k) = lammini
            n0i (k) = pow(lami(k),4)*qi3d(k)/cons12
            ni3d(k) = n0i(k)/lami(k)
          else if (lami(k) > lammaxi) then
            lami(k) = lammaxi
            n0i (k) = pow(lami(k),4)*qi3d(k)/cons12
            ni3d(k) = n0i(k)/lami(k)
          endif
        endif
        if (qr3d(k) >= qsmall) then
          lamr(k) = pow(pi*rhow*nr3d(k)/qr3d(k),1./3.)
          n0rr(k) = nr3d(k)*lamr(k)
          if (lamr(k) < lamminr) then
            lamr(k) = lamminr
            n0rr(k) = pow(lamr(k),4)*qr3d(k)/(pi*rhow)
            nr3d(k) = n0rr(k)/lamr(k)
          else if (lamr(k) > lammaxr) then
            lamr(k) = lammaxr
            n0rr(k) = pow(lamr(k),4)*qr3d(k)/(pi*rhow)
            nr3d(k) = n0rr(k)/lamr(k)
          endif
        endif
        if (qc3d(k) >= qsmall) then
          dum     = pres(k)/(287.15*t3d(k))
          pgam(k) = 0.0005714*(nc3d(k)/1.e6*dum)+0.2714
          pgam(k) = 1./(pow(pgam(k),2))-1.
          pgam(k) = max(pgam(k),2.)
          pgam(k) = min(pgam(k),10.)
          lamc(k) = pow(cons26*nc3d(k)*gamma(pgam(k)+4.)/(qc3d(k)*gamma(pgam(k)+1.)),1./3.)
          lammin  = (pgam(k)+1.)/60.e-6
          lammax  = (pgam(k)+1.)/1.e-6
          if (lamc(k) < lammin) then
            lamc(k) = lammin
            nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26
          else if (lamc(k) > lammax) then
            lamc(k) = lammax
            nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26
          endif
          cdist1(k) = nc3d(k)/gamma(pgam(k)+1.)
        endif
        if (qni3d(k) >= qsmall) then
          lams(k) = pow(cons1*ns3d(k)/qni3d(k),1./ds)
          n0s (k) = ns3d(k)*lams(k)
          if (lams(k) < lammins) then
            lams(k) = lammins
            n0s (k) = pow(lams(k),4)*qni3d(k)/cons1
            ns3d(k) = n0s(k)/lams(k)
          else if (lams(k) > lammaxs) then
            lams(k) = lammaxs
            n0s (k) = pow(lams(k),4)*qni3d(k)/cons1
            ns3d(k) = n0s(k)/lams(k)
          endif
        endif
        if (qg3d(k) >= qsmall) then
          lamg(k) = pow(cons2*ng3d(k)/qg3d(k),1./dg)
          n0g (k) = ng3d(k)*lamg(k)
          if (lamg(k) < lamming) then
            lamg(k) = lamming
            n0g (k) = pow(lamg(k),4)*qg3d(k)/cons2
            ng3d(k) = n0g(k)/lamg(k)
          else if (lamg(k) > lammaxg) then
            lamg(k) = lammaxg
            n0g (k) = pow(lamg(k),4)*qg3d(k)/cons2
            ng3d(k) = n0g(k)/lamg(k)
          endif
        endif
        mnuccc (k) = 0.
        nnuccc (k) = 0.
        prc    (k) = 0.
        nprc   (k) = 0.
        nprc1  (k) = 0.
        nsagg  (k) = 0.
        psacws (k) = 0.
        npsacws(k) = 0.
        psacwi (k) = 0.
        npsacwi(k) = 0.
        pracs  (k) = 0.
        npracs (k) = 0.
        nmults (k) = 0.
        qmults (k) = 0.
        nmultr (k) = 0.
        qmultr (k) = 0.
        nmultg (k) = 0.
        qmultg (k) = 0.
        nmultrg(k) = 0.
        qmultrg(k) = 0.
        mnuccr (k) = 0.
        nnuccr (k) = 0.
        pra    (k) = 0.
        npra   (k) = 0.
        nragg  (k) = 0.
        prci   (k) = 0.
        nprci  (k) = 0.
        prai   (k) = 0.
        nprai  (k) = 0.
        nnuccd (k) = 0.
        mnuccd (k) = 0.
        pcc    (k) = 0.
        pre    (k) = 0.
        prd    (k) = 0.
        prds   (k) = 0.
        eprd   (k) = 0.
        eprds  (k) = 0.
        nsubc  (k) = 0.
        nsubi  (k) = 0.
        nsubs  (k) = 0.
        nsubr  (k) = 0.
        piacr  (k) = 0.
        niacr  (k) = 0.
        praci  (k) = 0.
        piacrs (k) = 0.
        niacrs (k) = 0.
        pracis (k) = 0.
        pracg  (k) = 0.
        psacr  (k) = 0.
        psacwg (k) = 0.
        pgsacw (k) = 0.
        pgracs (k) = 0.
        prdg   (k) = 0.
        eprdg  (k) = 0.
        npracg (k) = 0.
        npsacwg(k) = 0.
        nscng  (k) = 0.
        ngracs (k) = 0.
        nsubg  (k) = 0.
        if (qc3d(k) >= qsmall .and. t3d(k) < 269.15) then
          nacnt     = exp(-2.80+0.262*(273.15-t3d(k)))*1000.
          dum       = 7.37*t3d(k)/(288.*10.*pres(k))/100.
          dap   (k) = cons37*t3d(k)*(1.+dum/rin)/mu(k)
          mnuccc(k) = cons38*dap(k)*nacnt*exp(log(cdist1(k))+log(gamma(pgam(k)+5.))-4.*log(lamc(k)))
          nnuccc(k) = 2.*pi*dap(k)*nacnt*cdist1(k)*gamma(pgam(k)+2.)/lamc(k)
          mnuccc(k) = mnuccc(k)+cons39*exp(log(cdist1(k))+log(gamma(7.+pgam(k)))-6.*log(lamc(k)))*(exp(aimm*(273.15-t3d(k)))-1.)
          nnuccc(k) = nnuccc(k)+cons40*exp(log(cdist1(k))+log(gamma(pgam(k)+4.))-3.*log(lamc(k)))*(exp(aimm*(273.15-t3d(k)))-1.)
          nnuccc(k) = min(nnuccc(k),nc3d(k)/dt)
        endif
        if (qc3d(k) >= 1.e-6) then
          prc  (k) = 1350.*pow(qc3d(k),2.47)*pow(nc3d(k)/1.e6*rho(k),-1.79)
          nprc1(k) = prc(k)/cons29
          nprc (k) = prc(k)/(qc3d(k)/nc3d(k))
          nprc (k) = min( nprc (k) , nc3d(k)/dt )
          nprc1(k) = min( nprc1(k) , nprc(k)    )
        endif
        if (qni3d(k) >= 1.e-8) then
          nsagg(k) = cons15*asn(k)*pow(rho(k),(2.+bs)/3.)*pow(qni3d(k),(2.+bs)/3.)*pow(ns3d(k)*rho(k),(4.-bs)/3.)/(rho(k))
        endif
        if (qni3d(k) >= 1.e-8 .and. qc3d(k) >= qsmall) then
          psacws (k) = cons13*asn(k)*qc3d(k)*rho(k)*n0s(k)/pow(lams(k),bs+3.)
          npsacws(k) = cons13*asn(k)*nc3d(k)*rho(k)*n0s(k)/pow(lams(k),bs+3.)
        endif
        if (qg3d(k) >= 1.e-8 .and. qc3d(k) >= qsmall) then
          psacwg (k) = cons14*agn(k)*qc3d(k)*rho(k)*n0g(k)/pow(lamg(k),bg+3.)
          npsacwg(k) = cons14*agn(k)*nc3d(k)*rho(k)*n0g(k)/pow(lamg(k),bg+3.)
        endif
        if (qi3d(k) >= 1.e-8 .and. qc3d(k) >= qsmall) then
          if (1./lami(k) >= 100.e-6) then
            psacwi (k) = cons16*ain(k)*qc3d(k)*rho(k)*n0i(k)/pow(lami(k),bi+3.)
            npsacwi(k) = cons16*ain(k)*nc3d(k)*rho(k)*n0i(k)/pow(lami(k),bi+3.)
          endif
        endif
        if (qr3d(k) >= 1.e-8.and.qni3d(k) >= 1.e-8) then
          ums = asn(k)*cons3/pow(lams(k),bs)
          umr = arn(k)*cons4/pow(lamr(k),br)
          uns = asn(k)*cons5/pow(lams(k),bs)
          unr = arn(k)*cons6/pow(lamr(k),br)
          dum = pow(rhosu/rho(k),0.54)
          ums = min( ums , 1.2*dum )
          uns = min( uns , 1.2*dum )
          umr = min( umr , 9.1*dum )
          unr = min( unr , 9.1*dum )
          pracs(k) = cons41*(pow(pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(k)*n0rr(k)*n0s(k)/pow(lamr(k),3)* &
                     (5./(pow(lamr(k),3)*lams(k))+2./(pow(lamr(k),2)*pow(lams(k),2))+0.5/(lamr(k)*pow(lams(k),3))))
          npracs(k) = cons32*rho(k)*pow(1.7*pow(unr-uns,2)+0.3*unr*uns,0.5)*n0rr(k)*n0s(k)*(1./(pow(lamr(k),3)*lams(k))+ &
                      1./(pow(lamr(k),2)*pow(lams(k),2))+1./(lamr(k)*pow(lams(k),3)))
          pracs(k) = min(pracs(k),qr3d(k)/dt)
          if (qni3d(k) >= 0.1e-3.and.qr3d(k) >= 0.1e-3) then
            psacr(k) = cons31*(pow(pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(k)*n0rr(k)*n0s(k)/pow(lams(k),3)* &
                       (5./(pow(lams(k),3)*lamr(k))+2./(pow(lams(k),2)*pow(lamr(k),2))+0.5/(lams(k)*pow(lamr(k),3))))            
          endif
        endif
        if (qr3d(k) >= 1.e-8.and.qg3d(k) >= 1.e-8) then
          umg = agn(k)*cons7/pow(lamg(k),bg)
          umr = arn(k)*cons4/pow(lamr(k),br)
          ung = agn(k)*cons8/pow(lamg(k),bg)
          unr = arn(k)*cons6/pow(lamr(k),br)
          dum = pow(rhosu/rho(k),0.54)
          umg = min( umg , 20.*dum )
          ung = min( ung , 20.*dum )
          umr = min( umr , 9.1*dum )
          unr = min( unr , 9.1*dum )
          pracg (k) = cons41*(pow(pow(1.2*umr-0.95*umg,2)+0.08*umg*umr,0.5)*rho(k)*n0rr(k)*n0g(k)/pow(lamr(k),3)* &
                      (5./(pow(lamr(k),3)*lamg(k))+2./(pow(lamr(k),2)*pow(lamg(k),2))+0.5/(lamr(k)*pow(lamg(k),3))))
          npracg(k) = cons32*rho(k)*pow(1.7*pow(unr-ung,2)+0.3*unr*ung,0.5)*n0rr(k)*n0g(k)*(1./(pow(lamr(k),3)*lamg(k))+ &
                      1./(pow(lamr(k),2)*pow(lamg(k),2))+1./(lamr(k)*pow(lamg(k),3)))
          pracg (k) = min(pracg(k),qr3d(k)/dt)
        endif
        if (qni3d(k) >= 0.1e-3) then
          if (qc3d(k) >= 0.5e-3.or.qr3d(k) >= 0.1e-3) then
            if (psacws(k) > 0..or.pracs(k) > 0.) then
              if (t3d(k) < 270.16 .and. t3d(k) > 265.16) then
                if (t3d(k) > 270.16) then
                  fmult = 0.
                else if (t3d(k) <= 270.16.and.t3d(k) > 268.16)  then
                  fmult = (270.16-t3d(k))/2.
                else if (t3d(k) >= 265.16.and.t3d(k) <= 268.16)   then
                  fmult = (t3d(k)-265.16)/3.
                else if (t3d(k) < 265.16) then
                  fmult = 0.
                endif
                if (psacws(k) > 0.) then
                  nmults(k) = 35.e4*psacws(k)*fmult*1000.
                  qmults(k) = nmults(k)*mmult
                  qmults(k) = min(qmults(k),psacws(k))
                  psacws(k) = psacws(k)-qmults(k)
                endif
                if (pracs(k) > 0.) then
                  nmultr(k) = 35.e4*pracs(k)*fmult*1000.
                  qmultr(k) = nmultr(k)*mmult
                  qmultr(k) = min(qmultr(k),pracs(k))
                  pracs(k) = pracs(k)-qmultr(k)
                endif
              endif
            endif
          endif
        endif
        if (qg3d(k) >= 0.1e-3) then
          if (qc3d(k) >= 0.5e-3.or.qr3d(k) >= 0.1e-3) then
            if (psacwg(k) > 0..or.pracg(k) > 0.) then
              if (t3d(k) < 270.16 .and. t3d(k) > 265.16) then
                if (t3d(k) > 270.16) then
                  fmult = 0.
                else if (t3d(k) <= 270.16.and.t3d(k) > 268.16)  then
                  fmult = (270.16-t3d(k))/2.
                else if (t3d(k) >= 265.16.and.t3d(k) <= 268.16)   then
                  fmult = (t3d(k)-265.16)/3.
                else if (t3d(k) < 265.16) then
                  fmult = 0.
                endif
                if (psacwg(k) > 0.) then
                  nmultg(k) = 35.e4*psacwg(k)*fmult*1000.
                  qmultg(k) = nmultg(k)*mmult
                  qmultg(k) = min(qmultg(k),psacwg(k))
                  psacwg(k) = psacwg(k)-qmultg(k)
                endif
                if (pracg(k) > 0.) then
                  nmultrg(k) = 35.e4*pracg(k)*fmult*1000.
                  qmultrg(k) = nmultrg(k)*mmult
                  qmultrg(k) = min(qmultrg(k),pracg(k))
                  pracg(k) = pracg(k)-qmultrg(k)
                endif
              endif
            endif
          endif
        endif
        if (psacws(k) > 0.) then
          if (qni3d(k) >= 0.1e-3.and.qc3d(k) >= 0.5e-3) then
            pgsacw(k) = min(psacws(k),cons17*dt*n0s(k)*qc3d(k)*qc3d(k)*asn(k)*asn(k)/(rho(k)*pow(lams(k),2.*bs+2.)))
            dum       = max(rhosn/(rhog-rhosn)*pgsacw(k),0.) 
            nscng (k) = dum/mg0*rho(k)
            nscng (k) = min(nscng(k),ns3d(k)/dt)
            psacws(k) = psacws(k) - pgsacw(k)
          endif
        endif
        if (pracs(k) > 0.) then
          if (qni3d(k) >= 0.1e-3.and.qr3d(k) >= 0.1e-3) then
            dum       = cons18*pow(4./lams(k),3)*pow(4./lams(k),3)/(cons18*pow(4./lams(k),3)*pow(4./lams(k),3)+ &  
                        cons19*pow(4./lamr(k),3)*pow(4./lamr(k),3))
            dum       = min( dum , 1. )
            dum       = max( dum , 0. )
            pgracs(k) = (1.-dum)*pracs(k)
            ngracs(k) = (1.-dum)*npracs(k)
            ngracs(k) = min(ngracs(k),nr3d(k)/dt)
            ngracs(k) = min(ngracs(k),ns3d(k)/dt)
            pracs (k) = pracs (k) - pgracs(k)
            npracs(k) = npracs(k) - ngracs(k)
            psacr (k) = psacr(k)*(1.-dum)
          endif
        endif
        if (t3d(k) < 269.15.and.qr3d(k) >= qsmall) then
          mnuccr(k) = cons20*nr3d(k)*(exp(aimm*(273.15-t3d(k)))-1.)/pow(lamr(k),3)/pow(lamr(k),3)
          nnuccr(k) = pi*nr3d(k)*bimm*(exp(aimm*(273.15-t3d(k)))-1.)/pow(lamr(k),3)
          nnuccr(k) = min(nnuccr(k),nr3d(k)/dt)
        endif
        if (qr3d(k) >= 1.e-8 .and. qc3d(k) >= 1.e-8) then
          dum     = (qc3d(k)*qr3d(k))
          pra (k) = 67.*pow(dum,1.15)
          npra(k) = pra(k)/(qc3d(k)/nc3d(k))
        endif
        if (qr3d(k) >= 1.e-8) then
          dum1=300.e-6
          if (1./lamr(k) < dum1) then
            dum=1.
          else if (1./lamr(k) >= dum1) then
            dum=2.-exp(2300.*(1./lamr(k)-dum1))
          endif
          nragg(k) = -5.78*dum*nr3d(k)*qr3d(k)*rho(k)
        endif
        if (qi3d(k) >= 1.e-8 .and.qvqvsi(k) >= 1.) then
          nprci(k) = cons21*(qv3d(k)-qvi(k))*rho(k)*n0i(k)*exp(-lami(k)*dcs)*dv(k)/abi(k)
          prci(k) = cons22*nprci(k)
          nprci(k) = min(nprci(k),ni3d(k)/dt)
        endif
        if (qni3d(k) >= 1.e-8 .and. qi3d(k) >= qsmall) then
          prai (k) = cons23*asn(k)*qi3d(k)*rho(k)*n0s(k)/pow(lams(k),bs+3.)
          nprai(k) = cons23*asn(k)*ni3d(k)*rho(k)*n0s(k)/pow(lams(k),bs+3.)
          nprai(k) = min( nprai(k) , ni3d(k)/dt )
        endif
        if (qr3d(k) >= 1.e-8 .and. qi3d(k) >= 1.e-8 .and. t3d(k) <= 273.15) then
          if (qr3d(k) >= 0.1e-3) then
            niacr(k)=cons24*ni3d(k)*n0rr(k)*arn(k)/pow(lamr(k),br+3.)*rho(k)
            piacr(k)=cons25*ni3d(k)*n0rr(k)*arn(k)/pow(lamr(k),br+3.)/pow(lamr(k),3)*rho(k)
            praci(k)=cons24*qi3d(k)*n0rr(k)*arn(k)/pow(lamr(k),br+3.)*rho(k)
            niacr(k)=min(niacr(k),nr3d(k)/dt)
            niacr(k)=min(niacr(k),ni3d(k)/dt)
          else 
            niacrs(k)=cons24*ni3d(k)*n0rr(k)*arn(k)/pow(lamr(k),br+3.)*rho(k)
            piacrs(k)=cons25*ni3d(k)*n0rr(k)*arn(k)/pow(lamr(k),br+3.)/pow(lamr(k),3)*rho(k)
            pracis(k)=cons24*qi3d(k)*n0rr(k)*arn(k)/pow(lamr(k),br+3.)*rho(k)
            niacrs(k)=min(niacrs(k),nr3d(k)/dt)
            niacrs(k)=min(niacrs(k),ni3d(k)/dt)
          endif
        endif
        if (inuc==0) then
          if ((qvqvs(k) >= 0.999 .and. t3d(k) <= 265.15) .or. qvqvsi(k) >= 1.08) then
            kc2 = 0.005*exp(0.304*(273.15-t3d(k)))*1000. ! convert from l-1 to m-3
            kc2 = min( kc2 ,500.e3 )
            kc2 = max( kc2/rho(k) , 0. )  ! convert to kg-1
            if (kc2 > ni3d(k)+ns3d(k)+ng3d(k)) then
              nnuccd(k) = (kc2-ni3d(k)-ns3d(k)-ng3d(k))/dt
              mnuccd(k) = nnuccd(k)*mi0
            endif
          endif
        else if (inuc==1) then
          if (t3d(k) < 273.15.and.qvqvsi(k) > 1.) then
            kc2 = 0.16*1000./rho(k)  ! convert from l-1 to kg-1
            if (kc2 > ni3d(k)+ns3d(k)+ng3d(k)) then
              nnuccd(k) = (kc2-ni3d(k)-ns3d(k)-ng3d(k))/dt
              mnuccd(k) = nnuccd(k)*mi0
            endif
          endif
        endif

        if (qi3d(k) >= qsmall) then
           epsi = 2.*pi*n0i(k)*rho(k)*dv(k)/(lami(k)*lami(k))
        else
           epsi = 0.
        endif
        if (qni3d(k) >= qsmall) then
          epss = 2.*pi*n0s(k)*rho(k)*dv(k)*(f1s/(lams(k)*lams(k))+f2s*pow(asn(k)*rho(k)/mu(k),0.5)*pow(sc(k),1./3.)*cons10/(pow(lams(k),cons35)))
        else
          epss = 0.
        endif
        if (qg3d(k) >= qsmall) then
          epsg = 2.*pi*n0g(k)*rho(k)*dv(k)*(f1s/(lamg(k)*lamg(k))+f2s*pow(agn(k)*rho(k)/mu(k),0.5)*pow(sc(k),1./3.)*cons11/(pow(lamg(k),cons36)))
        else
          epsg = 0.
        endif
        if (qr3d(k) >= qsmall) then
          epsr = 2.*pi*n0rr(k)*rho(k)*dv(k)*(f1r/(lamr(k)*lamr(k))+f2r*pow(arn(k)*rho(k)/mu(k),0.5)*pow(sc(k),1./3.)*cons9/(pow(lamr(k),cons34)))
        else
          epsr = 0.
        endif
        if (qi3d(k) >= qsmall) then              
          dum    = (1.-exp(-lami(k)*dcs)*(1.+lami(k)*dcs))
          prd(k) = epsi*(qv3d(k)-qvi(k))/abi(k)*dum
        else
          dum=0.
        endif
        if (qni3d(k) >= qsmall) then
          prds(k) = epss*(qv3d(k)-qvi(k))/abi(k)+epsi*(qv3d(k)-qvi(k))/abi(k)*(1.-dum)
        else
          prd(k) = prd(k)+epsi*(qv3d(k)-qvi(k))/abi(k)*(1.-dum)
        endif
        prdg(k) = epsg*(qv3d(k)-qvi(k))/abi(k)
        if (qv3d(k) < qvs(k)) then
          pre(k) = epsr*(qv3d(k)-qvs(k))/ab(k)
          pre(k) = min( pre(k) , 0. )
        else
          pre(k) = 0.
        endif
        dum = (qv3d(k)-qvi(k))/dt
        fudgef = 0.9999
        sum_dep = prd(k)+prds(k)+mnuccd(k)+prdg(k)
        if( (dum > 0. .and. sum_dep > dum*fudgef) .or. (dum < 0. .and. sum_dep < dum*fudgef) ) then
          mnuccd(k) = fudgef*mnuccd(k)*dum/sum_dep
          prd(k) = fudgef*prd(k)*dum/sum_dep
          prds(k) = fudgef*prds(k)*dum/sum_dep
          prdg(k) = fudgef*prdg(k)*dum/sum_dep
        endif
        if (prd(k) < 0.) then
          eprd(k)=prd(k)
          prd (k)=0.
        endif
        if (prds(k) < 0.) then
          eprds(k)=prds(k)
          prds (k)=0.
        endif
        if (prdg(k) < 0.) then
          eprdg(k)=prdg(k)
          prdg (k)=0.
        endif
        if (iliq==1) then
          mnuccc(k)=0.
          nnuccc(k)=0.
          mnuccr(k)=0.
          nnuccr(k)=0.
          mnuccd(k)=0.
          nnuccd(k)=0.
        endif
        if (igraup==1) then
          pracg  (k) = 0.
          psacr  (k) = 0.
          psacwg (k) = 0.
          prdg   (k) = 0.
          eprdg  (k) = 0.
          evpmg  (k) = 0.
          pgmlt  (k) = 0.
          npracg (k) = 0.
          npsacwg(k) = 0.
          nscng  (k) = 0.
          ngracs (k) = 0.
          nsubg  (k) = 0.
          ngmltg (k) = 0.
          ngmltr (k) = 0.
          piacrs (k) = piacrs(k)+piacr (k)
          piacr  (k) = 0.
          pracis (k) = pracis(k)+praci (k)
          praci  (k) = 0.
          psacws (k) = psacws(k)+pgsacw(k)
          pgsacw (k) = 0.
          pracs  (k) = pracs (k)+pgracs(k)
          pgracs (k) = 0.
        endif
        dum = (prc(k)+pra(k)+mnuccc(k)+psacws(k)+psacwi(k)+qmults(k)+psacwg(k)+pgsacw(k)+qmultg(k))*dt
        if (dum > qc3d(k) .and. qc3d(k) >= qsmall) then
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
        endif
        dum = (-prd(k)-mnuccc(k)+prci(k)+prai(k)-qmults(k)-qmultg(k)-qmultr(k)-qmultrg(k)-mnuccd(k)+praci(k)+pracis(k)-eprd(k)-psacwi(k))*dt
        if (dum > qi3d(k) .and. qi3d(k) >= qsmall) then
          ratio = (qi3d(k)/dt+prd(k)+mnuccc(k)+qmults(k)+qmultg(k)+qmultr(k)+qmultrg(k)+mnuccd(k)+psacwi(k))/(prci(k)+prai(k)+praci(k)+pracis(k)-eprd(k))
          prci(k) = prci(k)*ratio
          prai(k) = prai(k)*ratio
          praci(k) = praci(k)*ratio
          pracis(k) = pracis(k)*ratio
          eprd(k) = eprd(k)*ratio
        endif
        dum = ((pracs(k)-pre(k))+(qmultr(k)+qmultrg(k)-prc(k))+(mnuccr(k)-pra(k))+piacr(k)+piacrs(k)+pgracs(k)+pracg(k))*dt
        if (dum > qr3d(k).and.qr3d(k) >= qsmall) then
          ratio = (qr3d(k)/dt+prc(k)+pra(k))/(-pre(k)+qmultr(k)+qmultrg(k)+pracs(k)+mnuccr(k)+piacr(k)+piacrs(k)+pgracs(k)+pracg(k))
          pre(k) = pre(k)*ratio
          pracs(k) = pracs(k)*ratio
          qmultr(k) = qmultr(k)*ratio
          qmultrg(k) = qmultrg(k)*ratio
          mnuccr(k) = mnuccr(k)*ratio
          piacr(k) = piacr(k)*ratio
          piacrs(k) = piacrs(k)*ratio
          pgracs(k) = pgracs(k)*ratio
          pracg(k) = pracg(k)*ratio
        endif
        if (igraup==0) then
          dum = (-prds(k)-psacws(k)-prai(k)-prci(k)-pracs(k)-eprds(k)+psacr(k)-piacrs(k)-pracis(k))*dt
          if (dum > qni3d(k).and.qni3d(k) >= qsmall) then
            ratio = (qni3d(k)/dt+prds(k)+psacws(k)+prai(k)+prci(k)+pracs(k)+piacrs(k)+pracis(k))/(-eprds(k)+psacr(k))
            eprds(k) = eprds(k)*ratio
            psacr(k) = psacr(k)*ratio
          endif
        else if (igraup==1) then
          dum = (-prds(k)-psacws(k)-prai(k)-prci(k)-pracs(k)-eprds(k)+psacr(k)-piacrs(k)-pracis(k)-mnuccr(k))*dt
          if (dum > qni3d(k).and.qni3d(k) >= qsmall) then
            ratio = (qni3d(k)/dt+prds(k)+psacws(k)+prai(k)+prci(k)+pracs(k)+piacrs(k)+pracis(k)+mnuccr(k))/(-eprds(k)+psacr(k))
            eprds(k) = eprds(k)*ratio
            psacr(k) = psacr(k)*ratio
          endif
        endif
        dum = (-psacwg(k)-pracg(k)-pgsacw(k)-pgracs(k)-prdg(k)-mnuccr(k)-eprdg(k)-piacr(k)-praci(k)-psacr(k))*dt
        if (dum > qg3d(k).and.qg3d(k) >= qsmall) then
          ratio = (qg3d(k)/dt+psacwg(k)+pracg(k)+pgsacw(k)+pgracs(k)+prdg(k)+mnuccr(k)+psacr(k)+piacr(k)+praci(k))/(-eprdg(k))
          eprdg(k) = eprdg(k)*ratio
        endif
        qv3dten(k) = qv3dten(k)+(-pre(k)-prd(k)-prds(k)-mnuccd(k)-eprd(k)-eprds(k)-prdg(k)-eprdg(k))
        t3dten(k) = t3dten(k)+(pre(k)*xxlv(k)+(prd(k)+prds(k)+mnuccd(k)+eprd(k)+eprds(k)+prdg(k)+eprdg(k))*xxls(k)+ &
                    (psacws(k)+psacwi(k)+mnuccc(k)+mnuccr(k)+qmults(k)+qmultg(k)+qmultr(k)+qmultrg(k)+pracs(k) &
                     +psacwg(k)+pracg(k)+pgsacw(k)+pgracs(k)+piacr(k)+piacrs(k))*xlf(k))/cpm(k)
        qc3dten(k) = qc3dten(k)+(-pra(k)-prc(k)-mnuccc(k)+pcc(k)-psacws(k)-psacwi(k)-qmults(k)-qmultg(k)-psacwg(k)-pgsacw(k))
        qi3dten(k) = qi3dten(k)+(prd(k)+eprd(k)+psacwi(k)+mnuccc(k)-prci(k)- &
                     prai(k)+qmults(k)+qmultg(k)+qmultr(k)+qmultrg(k)+mnuccd(k)-praci(k)-pracis(k))
        qr3dten(k) = qr3dten(k)+(pre(k)+pra(k)+prc(k)-pracs(k)-mnuccr(k)-qmultr(k)-qmultrg(k) &
                     -piacr(k)-piacrs(k)-pracg(k)-pgracs(k))
        if (igraup==0) then
          qni3dten(k) = qni3dten(k)+(prai(k)+psacws(k)+prds(k)+pracs(k)+prci(k)+eprds(k)-psacr(k)+piacrs(k)+pracis(k))
          ns3dten(k) = ns3dten(k)+(nsagg(k)+nprci(k)-nscng(k)-ngracs(k)+niacrs(k))
          qg3dten(k) = qg3dten(k)+(pracg(k)+psacwg(k)+pgsacw(k)+pgracs(k)+prdg(k)+eprdg(k)+mnuccr(k)+piacr(k)+praci(k)+psacr(k))
          ng3dten(k) = ng3dten(k)+(nscng(k)+ngracs(k)+nnuccr(k)+niacr(k))
        else if (igraup==1) then
          qni3dten(k) = qni3dten(k)+(prai(k)+psacws(k)+prds(k)+pracs(k)+prci(k)+eprds(k)-psacr(k)+piacrs(k)+pracis(k)+mnuccr(k))
          ns3dten(k) = ns3dten(k)+(nsagg(k)+nprci(k)-nscng(k)-ngracs(k)+niacrs(k)+nnuccr(k))
        endif
        nc3dten(k) = nc3dten(k)+(-nnuccc(k)-npsacws(k)-npra(k)-nprc(k)-npsacwi(k)-npsacwg(k))
        ni3dten(k) = ni3dten(k)+(nnuccc(k)-nprci(k)-nprai(k)+nmults(k)+nmultg(k)+nmultr(k)+nmultrg(k)+nnuccd(k)-niacr(k)-niacrs(k))
        nr3dten(k) = nr3dten(k)+(nprc1(k)-npracs(k)-nnuccr(k)+nragg(k)-niacr(k)-niacrs(k)-npracg(k)-ngracs(k))
        c2prec (k) = pra(k)+prc(k)+psacws(k)+qmults(k)+qmultg(k)+psacwg(k)+pgsacw(k)+mnuccc(k)+psacwi(k)
        dumt       = t3d(k)+dt*t3dten(k)
        dumqv      = qv3d(k)+dt*qv3dten(k)
        dum        = min( 0.99*pres(k) , polysvp(dumt,0) )
        dumqss     = ep_2*dum/(pres(k)-dum)
        dumqc      = qc3d(k)+dt*qc3dten(k)
        dumqc      = max( dumqc , 0. )
        dums       = dumqv-dumqss
        pcc(k)     = dums/(1.+pow(xxlv(k),2)*dumqss/(cpm(k)*rv*pow(dumt,2)))/dt
        if (pcc(k)*dt+dumqc < 0.) pcc(k) = -dumqc/dt
        qv3dten(k) = qv3dten(k)-pcc(k)
        t3dten (k) = t3dten (k)+pcc(k)*xxlv(k)/cpm(k)
        qc3dten(k) = qc3dten(k)+pcc(k)
        if (eprd(k) < 0.) then
          dum      = eprd(k)*dt/qi3d(k)
          dum      = max(-1.,dum)
          nsubi(k) = dum*ni3d(k)/dt
        endif
        if (eprds(k) < 0.) then
          dum      = eprds(k)*dt/qni3d(k)
          dum      = max(-1.,dum)
          nsubs(k) = dum*ns3d(k)/dt
        endif
        if (pre(k) < 0.) then
          dum      = pre(k)*dt/qr3d(k)
          dum      = max(-1.,dum)
          nsubr(k) = dum*nr3d(k)/dt
        endif
        if (eprdg(k) < 0.) then
          dum      = eprdg(k)*dt/qg3d(k)
          dum      = max(-1.,dum)
          nsubg(k) = dum*ng3d(k)/dt
        endif
        ni3dten(k) = ni3dten(k)+nsubi(k)
        ns3dten(k) = ns3dten(k)+nsubs(k)
        ng3dten(k) = ng3dten(k)+nsubg(k)
        nr3dten(k) = nr3dten(k)+nsubr(k)
      endif !!!!!! temperature
      ltrue = 1
    enddo

    precrt  = 0.
    snowrt  = 0.
    snowprt = 0.
    grplprt = 0.

    if (ltrue==0) return
      nstep = 1
      do k = kte,kts,-1
        dumi  (k) = qi3d (k)+qi3dten (k)*dt
        dumqs (k) = qni3d(k)+qni3dten(k)*dt
        dumr  (k) = qr3d (k)+qr3dten (k)*dt
        dumfni(k) = ni3d (k)+ni3dten (k)*dt
        dumfns(k) = ns3d (k)+ns3dten (k)*dt
        dumfnr(k) = nr3d (k)+nr3dten (k)*dt
        dumc  (k) = qc3d (k)+qc3dten (k)*dt
        dumfnc(k) = nc3d (k)+nc3dten (k)*dt
        dumg  (k) = qg3d (k)+qg3dten (k)*dt
        dumfng(k) = ng3d (k)+ng3dten (k)*dt
        if (iinum==1) dumfnc(k) = nc3d(k)
        dumfni(k) = max( 0. , dumfni(k) )
        dumfns(k) = max( 0. , dumfns(k) )
        dumfnc(k) = max( 0. , dumfnc(k) )
        dumfnr(k) = max( 0. , dumfnr(k) )
        dumfng(k) = max( 0. , dumfng(k) )
        if (dumi(k) >= qsmall) then
          dlami = pow(cons12*dumfni(k)/dumi(k),1./di)
          dlami = max( dlami , lammini )
          dlami = min( dlami , lammaxi )
        endif
        if (dumr(k) >= qsmall) then
          dlamr = pow(pi*rhow*dumfnr(k)/dumr(k),1./3.)
          dlamr = max( dlamr , lamminr )
          dlamr = min( dlamr , lammaxr )
        endif
        if (dumc(k) >= qsmall) then
          dum     = pres(k)/(287.15*t3d(k))
          pgam(k) = 0.0005714*(nc3d(k)/1.e6*dum)+0.2714
          pgam(k) = 1./(pow(pgam(k),2))-1.
          pgam(k) = max(pgam(k),2.)
          pgam(k) = min(pgam(k),10.)
          dlamc   = pow(cons26*dumfnc(k)*gamma(pgam(k)+4.)/(dumc(k)*gamma(pgam(k)+1.)),1./3.)
          lammin  = (pgam(k)+1.)/60.e-6
          lammax  = (pgam(k)+1.)/1.e-6
          dlamc   = max(dlamc,lammin)
          dlamc   = min(dlamc,lammax)
        endif
        if (dumqs(k) >= qsmall) then
          dlams = pow(cons1*dumfns(k)/ dumqs(k),1./ds)
          dlams=max(dlams,lammins)
          dlams=min(dlams,lammaxs)
        endif
        if (dumg(k) >= qsmall) then
          dlamg = pow(cons2*dumfng(k)/ dumg(k),1./dg)
          dlamg=max(dlamg,lamming)
          dlamg=min(dlamg,lammaxg)
        endif
        if (dumc(k) >= qsmall) then
          unc =  acn(k)*gamma(1.+bc+pgam(k))/ (pow(dlamc,bc)*gamma(pgam(k)+1.))
          umc = acn(k)*gamma(4.+bc+pgam(k))/  (pow(dlamc,bc)*gamma(pgam(k)+4.))
        else
          umc = 0.
          unc = 0.
        endif
        if (dumi(k) >= qsmall) then
          uni = ain(k)*cons27/pow(dlami,bi)
          umi = ain(k)*cons28/pow(dlami,bi)
        else
          umi = 0.
          uni = 0.
        endif
        if (dumr(k) >= qsmall) then
          unr = arn(k)*cons6/pow(dlamr,br)
          umr = arn(k)*cons4/pow(dlamr,br)
        else
          umr = 0.
          unr = 0.
        endif
        if (dumqs(k) >= qsmall) then
          ums = asn(k)*cons3/pow(dlams,bs)
          uns = asn(k)*cons5/pow(dlams,bs)
        else
          ums = 0.
          uns = 0.
        endif
        if (dumg(k) >= qsmall) then
          umg = agn(k)*cons7/pow(dlamg,bg)
          ung = agn(k)*cons8/pow(dlamg,bg)
        else
          umg = 0.
          ung = 0.
        endif
        dum    = pow(rhosu/rho(k),0.54)
        ums    = min(ums,1.2*dum)
        uns    = min(uns,1.2*dum)
        umi    = min(umi,1.2*pow(rhosu/rho(k),0.35))
        uni    = min(uni,1.2*pow(rhosu/rho(k),0.35))
        umr    = min(umr,9.1*dum)
        unr    = min(unr,9.1*dum)
        umg    = min(umg,20.*dum)
        ung    = min(ung,20.*dum)
        fr (k) = umr
        fi (k) = umi
        fni(k) = uni
        fs (k) = ums
        fns(k) = uns
        fnr(k) = unr
        fc (k) = umc
        fnc(k) = unc
        fg (k) = umg
        fng(k) = ung
        if (k <= kte-1) then
          if (fr (k) < 1.e-10) fr (k) = fr (k+1)
          if (fi (k) < 1.e-10) fi (k) = fi (k+1)
          if (fni(k) < 1.e-10) fni(k) = fni(k+1)
          if (fs (k) < 1.e-10) fs (k) = fs (k+1)
          if (fns(k) < 1.e-10) fns(k) = fns(k+1)
          if (fnr(k) < 1.e-10) fnr(k) = fnr(k+1)
          if (fc (k) < 1.e-10) fc (k) = fc (k+1)
          if (fnc(k) < 1.e-10) fnc(k) = fnc(k+1)
          if (fg (k) < 1.e-10) fg (k) = fg (k+1)
          if (fng(k) < 1.e-10) fng(k) = fng(k+1)
        endif ! k le kte-1
        rgvm = max(fr(k),fi(k),fs(k),fc(k),fni(k),fnr(k),fns(k),fnc(k),fg(k),fng(k))
        nstep = max(int(rgvm*dt/dzq(k)+1.),nstep)
        dumr  (k) = dumr  (k)*rho(k)
        dumi  (k) = dumi  (k)*rho(k)
        dumfni(k) = dumfni(k)*rho(k)
        dumqs (k) = dumqs (k)*rho(k)
        dumfns(k) = dumfns(k)*rho(k)
        dumfnr(k) = dumfnr(k)*rho(k)
        dumc  (k) = dumc  (k)*rho(k)
        dumfnc(k) = dumfnc(k)*rho(k)
        dumg  (k) = dumg  (k)*rho(k)
        dumfng(k) = dumfng(k)*rho(k)
      enddo

      do n = 1,nstep
        do k = kts,kte
          faloutr (k) = fr (k)*dumr  (k)
          falouti (k) = fi (k)*dumi  (k)
          faloutni(k) = fni(k)*dumfni(k)
          falouts (k) = fs (k)*dumqs (k)
          faloutns(k) = fns(k)*dumfns(k)
          faloutnr(k) = fnr(k)*dumfnr(k)
          faloutc (k) = fc (k)*dumc  (k)
          faloutnc(k) = fnc(k)*dumfnc(k)
          faloutg (k) = fg (k)*dumg  (k)
          faloutng(k) = fng(k)*dumfng(k)
        enddo
        k        = kte
        faltndr  = faloutr (k)/dzq(k)
        faltndi  = falouti (k)/dzq(k)
        faltndni = faloutni(k)/dzq(k)
        faltnds  = falouts (k)/dzq(k)
        faltndns = faloutns(k)/dzq(k)
        faltndnr = faloutnr(k)/dzq(k)
        faltndc  = faloutc (k)/dzq(k)
        faltndnc = faloutnc(k)/dzq(k)
        faltndg  = faloutg (k)/dzq(k)
        faltndng = faloutng(k)/dzq(k)
        qrsten (k) = qrsten (k)-faltndr /nstep/rho(k)
        qisten (k) = qisten (k)-faltndi /nstep/rho(k)
        ni3dten(k) = ni3dten(k)-faltndni/nstep/rho(k)
        qnisten(k) = qnisten(k)-faltnds /nstep/rho(k)
        ns3dten(k) = ns3dten(k)-faltndns/nstep/rho(k)
        nr3dten(k) = nr3dten(k)-faltndnr/nstep/rho(k)
        qcsten (k) = qcsten (k)-faltndc /nstep/rho(k)
        nc3dten(k) = nc3dten(k)-faltndnc/nstep/rho(k)
        qgsten (k) = qgsten (k)-faltndg /nstep/rho(k)
        ng3dten(k) = ng3dten(k)-faltndng/nstep/rho(k)
        dumr  (k) = dumr  (k)-faltndr *dt/nstep
        dumi  (k) = dumi  (k)-faltndi *dt/nstep
        dumfni(k) = dumfni(k)-faltndni*dt/nstep
        dumqs (k) = dumqs (k)-faltnds *dt/nstep
        dumfns(k) = dumfns(k)-faltndns*dt/nstep
        dumfnr(k) = dumfnr(k)-faltndnr*dt/nstep
        dumc  (k) = dumc  (k)-faltndc *dt/nstep
        dumfnc(k) = dumfnc(k)-faltndnc*dt/nstep
        dumg  (k) = dumg  (k)-faltndg *dt/nstep
        dumfng(k) = dumfng(k)-faltndng*dt/nstep
        do k = kte-1,kts,-1
          faltndr  = (faloutr (k+1)-faloutr (k))/dzq(k)
          faltndi  = (falouti (k+1)-falouti (k))/dzq(k)
          faltndni = (faloutni(k+1)-faloutni(k))/dzq(k)
          faltnds  = (falouts (k+1)-falouts (k))/dzq(k)
          faltndns = (faloutns(k+1)-faloutns(k))/dzq(k)
          faltndnr = (faloutnr(k+1)-faloutnr(k))/dzq(k)
          faltndc  = (faloutc (k+1)-faloutc (k))/dzq(k)
          faltndnc = (faloutnc(k+1)-faloutnc(k))/dzq(k)
          faltndg  = (faloutg (k+1)-faloutg (k))/dzq(k)
          faltndng = (faloutng(k+1)-faloutng(k))/dzq(k)
          qrsten (k) = qrsten (k)+faltndr /nstep/rho(k)
          qisten (k) = qisten (k)+faltndi /nstep/rho(k)
          ni3dten(k) = ni3dten(k)+faltndni/nstep/rho(k)
          qnisten(k) = qnisten(k)+faltnds /nstep/rho(k)
          ns3dten(k) = ns3dten(k)+faltndns/nstep/rho(k)
          nr3dten(k) = nr3dten(k)+faltndnr/nstep/rho(k)
          qcsten (k) = qcsten (k)+faltndc /nstep/rho(k)
          nc3dten(k) = nc3dten(k)+faltndnc/nstep/rho(k)
          qgsten (k) = qgsten (k)+faltndg /nstep/rho(k)
          ng3dten(k) = ng3dten(k)+faltndng/nstep/rho(k)
          dumr  (k) = dumr  (k)+faltndr *dt/nstep
          dumi  (k) = dumi  (k)+faltndi *dt/nstep
          dumfni(k) = dumfni(k)+faltndni*dt/nstep
          dumqs (k) = dumqs (k)+faltnds *dt/nstep
          dumfns(k) = dumfns(k)+faltndns*dt/nstep
          dumfnr(k) = dumfnr(k)+faltndnr*dt/nstep
          dumc  (k) = dumc  (k)+faltndc *dt/nstep
          dumfnc(k) = dumfnc(k)+faltndnc*dt/nstep
          dumg  (k) = dumg  (k)+faltndg *dt/nstep
          dumfng(k) = dumfng(k)+faltndng*dt/nstep
          csed(k)=csed(k)+faloutc(k)/nstep
          ised(k)=ised(k)+falouti(k)/nstep
          ssed(k)=ssed(k)+falouts(k)/nstep
          gsed(k)=gsed(k)+faloutg(k)/nstep
          rsed(k)=rsed(k)+faloutr(k)/nstep
        enddo
        precrt  = precrt +(faloutr(kts)+faloutc(kts)+falouts(kts)+falouti(kts)+faloutg(kts))*dt/nstep
        snowrt  = snowrt +(falouts(kts)+falouti(kts)+faloutg(kts))*dt/nstep
        snowprt = snowprt+(falouti(kts)+falouts(kts))*dt/nstep
        grplprt = grplprt+(faloutg(kts))*dt/nstep
      enddo ! nstep

      do k=kts,kte
        qr3dten (k) = qr3dten (k) + qrsten (k)
        qi3dten (k) = qi3dten (k) + qisten (k)
        qc3dten (k) = qc3dten (k) + qcsten (k)
        qg3dten (k) = qg3dten (k) + qgsten (k)
        qni3dten(k) = qni3dten(k) + qnisten(k)
        if (qi3d(k) >= qsmall.and.t3d(k) < 273.15.and.lami(k) >= 1.e-10) then
          if (1./lami(k) >= 2.*dcs) then
            qni3dten(k) = qni3dten(k)+qi3d(k)/dt+ qi3dten(k)
            ns3dten(k) = ns3dten(k)+ni3d(k)/dt+   ni3dten(k)
            qi3dten(k) = -qi3d(k)/dt
            ni3dten(k) = -ni3d(k)/dt
          endif
        endif
        qc3d (k) = qc3d (k)+qc3dten (k)*dt
        qi3d (k) = qi3d (k)+qi3dten (k)*dt
        qni3d(k) = qni3d(k)+qni3dten(k)*dt
        qr3d (k) = qr3d (k)+qr3dten (k)*dt
        nc3d (k) = nc3d (k)+nc3dten (k)*dt
        ni3d (k) = ni3d (k)+ni3dten (k)*dt
        ns3d (k) = ns3d (k)+ns3dten (k)*dt
        nr3d (k) = nr3d (k)+nr3dten (k)*dt
        if (igraup==0) then
          qg3d(k) = qg3d(k)+qg3dten(k)*dt
          ng3d(k) = ng3d(k)+ng3dten(k)*dt
        endif
        t3d (k) = t3d (k)+t3dten (k)*dt
        qv3d(k) = qv3d(k)+qv3dten(k)*dt
        evs (k) = min( 0.99*pres(k) , polysvp(t3d(k),0) )   ! pa
        eis (k) = min( 0.99*pres(k) , polysvp(t3d(k),1) )   ! pa
        if (eis(k) > evs(k)) eis(k) = evs(k)
        qvs   (k) = ep_2*evs(k)/(pres(k)-evs(k))
        qvi   (k) = ep_2*eis(k)/(pres(k)-eis(k))
        qvqvs (k) = qv3d(k)/qvs(k)
        qvqvsi(k) = qv3d(k)/qvi(k)
        if (qvqvs(k) < 0.9) then
          if (qr3d(k) < 1.e-8) then
            qv3d(k)=qv3d(k)+qr3d(k)
            t3d (k)=t3d (k)-qr3d(k)*xxlv(k)/cpm(k)
            qr3d(k)=0.
          endif
          if (qc3d(k) < 1.e-8) then
            qv3d(k)=qv3d(k)+qc3d(k)
            t3d (k)=t3d (k)-qc3d(k)*xxlv(k)/cpm(k)
            qc3d(k)=0.
          endif
        endif
        if (qvqvsi(k) < 0.9) then
          if (qi3d(k) < 1.e-8) then
            qv3d(k)=qv3d(k)+qi3d(k)
            t3d (k)=t3d (k)-qi3d(k)*xxls(k)/cpm(k)
            qi3d(k)=0.
          endif
          if (qni3d(k) < 1.e-8) then
            qv3d (k)=qv3d(k)+qni3d(k)
            t3d  (k)=t3d (k)-qni3d(k)*xxls(k)/cpm(k)
            qni3d(k)=0.
          endif
          if (qg3d(k) < 1.e-8) then
            qv3d(k)=qv3d(k)+qg3d(k)
            t3d (k)=t3d (k)-qg3d(k)*xxls(k)/cpm(k)
            qg3d(k)=0.
          endif
        endif
        if (qc3d(k) < qsmall) then
          qc3d(k) = 0.
          nc3d(k) = 0.
          effc(k) = 0.
        endif
        if (qr3d(k) < qsmall) then
          qr3d(k) = 0.
          nr3d(k) = 0.
          effr(k) = 0.
        endif
        if (qi3d(k) < qsmall) then
          qi3d(k) = 0.
          ni3d(k) = 0.
          effi(k) = 0.
        endif
        if (qni3d(k) < qsmall) then
          qni3d(k) = 0.
          ns3d (k) = 0.
          effs (k) = 0.
        endif
        if (qg3d(k) < qsmall) then
          qg3d(k) = 0.
          ng3d(k) = 0.
          effg(k) = 0.
        endif
        if (.not. (qc3d(k) < qsmall.and.qi3d(k) < qsmall.and.qni3d(k) < qsmall .and.qr3d(k) < qsmall.and.qg3d(k) < qsmall)) then
          if (qi3d(k) >= qsmall.and.t3d(k) >= 273.15) then
            qr3d(k) = qr3d(k)+qi3d(k)
            t3d(k) = t3d(k)-qi3d(k)*xlf(k)/cpm(k)
            qi3d(k) = 0.
            nr3d(k) = nr3d(k)+ni3d(k)
            ni3d(k) = 0.
          endif
          if (iliq /= 1) then
            if (t3d(k) <= 233.15.and.qc3d(k) >= qsmall) then
              qi3d(k)=qi3d(k)+qc3d(k)
              t3d (k)=t3d (k)+qc3d(k)*xlf(k)/cpm(k)
              qc3d(k)=0.
              ni3d(k)=ni3d(k)+nc3d(k)
              nc3d(k)=0.
            endif
            if (igraup==0) then
              if (t3d(k) <= 233.15.and.qr3d(k) >= qsmall) then
                 qg3d(k) = qg3d(k)+qr3d(k)
                 t3d (k) = t3d (k)+qr3d(k)*xlf(k)/cpm(k)
                 qr3d(k) = 0.
                 ng3d(k) = ng3d(k)+ nr3d(k)
                 nr3d(k) = 0.
              endif
            else if (igraup==1) then
              if (t3d(k) <= 233.15.and.qr3d(k) >= qsmall) then
                qni3d(k) = qni3d(k)+qr3d(k)
                t3d  (k) = t3d  (k)+qr3d(k)*xlf(k)/cpm(k)
                qr3d (k) = 0.
                ns3d (k) = ns3d (k)+nr3d(k)
                nr3d (k) = 0.
              endif
            endif
          endif
          ni3d(k) = max( 0. , ni3d(k) )
          ns3d(k) = max( 0. , ns3d(k) )
          nc3d(k) = max( 0. , nc3d(k) )
          nr3d(k) = max( 0. , nr3d(k) )
          ng3d(k) = max( 0. , ng3d(k) )
          if (qi3d(k) >= qsmall) then
            lami(k) = pow(cons12*ni3d(k)/qi3d(k),1./di)
            if (lami(k) < lammini) then
              lami(k) = lammini
              n0i (k) = pow(lami(k),4)*qi3d(k)/cons12
              ni3d(k) = n0i (k)/lami(k)
            else if (lami(k) > lammaxi) then
              lami(k) = lammaxi
              n0i (k) = pow(lami(k),4)*qi3d(k)/cons12
              ni3d(k) = n0i (k)/lami(k)
            endif
          endif
        if (qr3d(k) >= qsmall) then
          lamr(k) = pow(pi*rhow*nr3d(k)/qr3d(k),1./3.)
          if (lamr(k) < lamminr) then
            lamr(k) = lamminr
            n0rr(k) = pow(lamr(k),4)*qr3d(k)/(pi*rhow)
            nr3d(k) = n0rr(k)/lamr(k)
          else if (lamr(k) > lammaxr) then
            lamr(k) = lammaxr
            n0rr(k) = pow(lamr(k),4)*qr3d(k)/(pi*rhow)
            nr3d(k) = n0rr(k)/lamr(k)
          endif
        endif
        if (qc3d(k) >= qsmall) then
          dum = pres(k)/(287.15*t3d(k))
          pgam(k)=0.0005714*(nc3d(k)/1.e6*dum)+0.2714
          pgam(k)=1./(pow(pgam(k),2))-1.
          pgam(k)=max(pgam(k),2.)
          pgam(k)=min(pgam(k),10.)
          lamc(k) = pow(cons26*nc3d(k)*gamma(pgam(k)+4.)/(qc3d(k)*gamma(pgam(k)+1.)),1./3.)
          lammin = (pgam(k)+1.)/60.e-6
          lammax = (pgam(k)+1.)/1.e-6
          if (lamc(k) < lammin) then
            lamc(k) = lammin
            nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26
          else if (lamc(k) > lammax) then
            lamc(k) = lammax
            nc3d(k) = exp(3.*log(lamc(k))+log(qc3d(k))+log(gamma(pgam(k)+1.))-log(gamma(pgam(k)+4.)))/cons26
          endif
        endif
        if (qni3d(k) >= qsmall) then
          lams(k) = pow(cons1*ns3d(k)/qni3d(k),1./ds)
          if (lams(k) < lammins) then
            lams(k) = lammins
            n0s (k) = pow(lams(k),4)*qni3d(k)/cons1
            ns3d(k) = n0s (k)/lams(k)
          else if (lams(k) > lammaxs) then
            lams(k) = lammaxs
            n0s (k) = pow(lams(k),4)*qni3d(k)/cons1
            ns3d(k) = n0s (k)/lams(k)
          endif
        endif
        if (qg3d(k) >= qsmall) then
          lamg(k) = pow(cons2*ng3d(k)/qg3d(k),1./dg)
          if (lamg(k) < lamming) then
            lamg(k) = lamming
            n0g (k) = pow(lamg(k),4)*qg3d(k)/cons2
            ng3d(k) = n0g (k)/lamg(k)
          else if (lamg(k) > lammaxg) then
            lamg(k) = lammaxg
            n0g (k) = pow(lamg(k),4)*qg3d(k)/cons2
            ng3d(k) = n0g (k)/lamg(k)
          endif
        endif
      endif
      if (qi3d(k) >= qsmall) then
        effi(k) = 3./lami(k)/2.*1.e6
      else
        effi(k) = 25.
      endif
      if (qni3d(k) >= qsmall) then
        effs(k) = 3./lams(k)/2.*1.e6
      else
        effs(k) = 25.
      endif
      if (qr3d(k) >= qsmall) then
        effr(k) = 3./lamr(k)/2.*1.e6
      else
        effr(k) = 25.
      endif
      if (qc3d(k) >= qsmall) then
        effc(k) = gamma(pgam(k)+4.)/gamma(pgam(k)+3.)/lamc(k)/2.*1.e6
      else
        effc(k) = 25.
      endif
      if (qg3d(k) >= qsmall) then
        effg(k) = 3./lamg(k)/2.*1.e6
      else
        effg(k) = 25.
      endif
      ni3d(k) = min( ni3d(k) , 0.3e6/rho(k) )
      if (iinum==0.and.iact.eq.2) then
        nc3d(k) = min( nc3d(k) , (nanew1+nanew2)/rho(k) )
      endif
      if (iinum==1) then 
        nc3d(k) = ndcnst*1.e6/rho(k)
      endif
    enddo !!! k loop
  end subroutine morr_two_moment_micro



  ! compute saturation vapor pressure
  ! polysvp returned in units of pa.
  ! t is input in units of k.
  ! type refers to saturation with respect to liquid (0) or ice (1)
  ! replace goff-gratch with faster formulation from flatau et al. 1992, table 4 (right-hand column)
  real function polysvp (t,type)
    implicit none
    real    :: dum
    real    :: t
    integer :: type
    ! ice
    real :: a0i,a1i,a2i,a3i,a4i,a5i,a6i,a7i,a8i 
    real :: a0,a1,a2,a3,a4,a5,a6,a7,a8 
    real :: dt
    a0 = 6.11239921
    a1 = 0.443987641
    a2 = 0.142986287e-1
    a3 = 0.264847430e-3
    a4 = 0.302950461e-5
    a5 = 0.206739458e-7
    a6 = 0.640689451e-10
    a7 = -0.952447341e-13
    a8 = -0.976195544e-15
    a0i = 6.11147274
    a1i = 0.503160820
    a2i = 0.188439774e-1
    a3i = 0.420895665e-3
    a4i = 0.615021634e-5
    a5i = 0.602588177e-7
    a6i = 0.385852041e-9
    a7i = 0.146898966e-11
    a8i = 0.252751365e-14
    ! ice
    if (type==1) then
      if (t >= 195.8) then
        dt=t-273.15
        polysvp = a0i + dt*(a1i+dt*(a2i+dt*(a3i+dt*(a4i+dt*(a5i+dt*(a6i+dt*(a7i+a8i*dt))))))) 
        polysvp = polysvp*100.
      else
        polysvp = pow(10.,-9.09718*(273.16/t-1.)-3.56654*alog10(273.16/t)+0.876793*(1.-t/273.16)+alog10(6.1071))*100.
      endif
    endif
    ! liquid
    if (type==0) then
      if (t >= 202.0) then
        dt = t-273.15
        polysvp = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))))
        polysvp = polysvp*100.
      else
        polysvp = pow(10.,-7.90298*(373.16/t-1.)+5.02808*alog10(373.16/t)-1.3816e-7*(pow(10.,11.344*(1.-t/373.16))-1.)+ &
                  8.1328e-3*(pow(10.,-3.49149*(373.16/t-1.))-1.)+alog10(1013.246))*100.                      
      endif
    endif
  end function polysvp



  real function gamma(x)
    implicit none
    integer :: i,n
    logical :: parity
    real    :: conv,eps,fact,res,sum,x,xbig,xden,xinf,xminin,xnum,y,y1,ysq,z
    real, dimension(7) :: c
    real, dimension(8) :: p
    real, dimension(8) :: q
    xbig   = 35.040e0
    xminin = 1.18e-38
    eps    = 1.19e-7
    xinf   = 3.4e38
    q(1) = -3.08402300119738975254353e+1
    q(2) =  3.15350626979604161529144e+2
    q(3) = -1.01515636749021914166146e+3
    q(4) = -3.10777167157231109440444e+3
    q(5) =  2.25381184209801510330112e+4
    q(6) =  4.75584627752788110767815e+3
    q(7) = -1.34659959864969306392456e+5
    q(8) = -1.15132259675553483497211e+5
    c(1) = -1.910444077728e-03
    c(2) =  8.4171387781295e-04
    c(3) = -5.952379913043012e-04
    c(4) =  7.93650793500350248e-04
    c(5) = -2.777777777777681622553e-03
    c(6) =  8.333333333333333331554247e-02
    c(7) =  5.7083835261e-03
    p(1) = -1.71618513886549492533811e+0
    p(2) =  2.47656508055759199108314e+1
    p(3) = -3.79804256470945635097577e+2
    p(4) =  6.29331155312818442661052e+2
    p(5) =  8.66966202790413211295064e+2
    p(6) = -3.14512729688483675254357e+4
    p(7) = -3.61444134186911729807069e+4
    p(8) =  6.64561438202405440627855e+4
    parity  = .false.
    fact    = 1.
    n       = 0
    y       = x
    if (y <= 0) then
      y   = -x
      y1  = aint(y)
      res = y-y1
      if (res /= 0) then
        if (y1 /= aint(y1*0.5)*2) parity=.true.
        fact = -pi/sin(pi*res)
        y    = y+1.
      else
        res=xinf
        gamma = res
        return
      endif
    endif
    if (y < eps) then
      if (y >= xminin) then
        res=1./y
      else
        res=xinf
        gamma = res
        return
      endif
    else if (y < 12) then
      y1 = y
      if(y < 1.) then
        z = y
        y = y+1.
      else
        n = int(y)-1
        y = y-real(n)
        z = y-1.
      endif
      xnum = 0
      xden = 1.
      do i=1,8
        xnum = (xnum+p(i))*z
        xden = xden*z+q(i)
      enddo
      res = xnum/xden+1.
      if (y1 < y) then
        res = res/y1
      else if (y1 > y) then
        do i=1,n
          res = res*y
          y   = y+1.
        enddo
      endif
    else
      if(y <= xbig)then
        ysq = y*y
        sum = c(7)
        do i=1,6
          sum = sum/ysq+c(i)
        enddo
        sum = sum/y-y+xxx
        sum = sum+(y-0.5)*log(y)
        res = exp(sum)
      else
        res = xinf
        gamma = res
        return
      endif
    endif
    if (parity) res=-res
    if (fact /= 1.) res=fact/res
    gamma = res
  end function gamma



  real function pow_rr(a,b)
    real, intent(in) :: a, b
    pow_rr = a**b
  end function pow_rr
  real function pow_ri(a,b)
    real   , intent(in) :: a
    integer, intent(in) :: b
    pow_ri = a**b
  end function pow_ri


end module module_mp_morr_two_moment

