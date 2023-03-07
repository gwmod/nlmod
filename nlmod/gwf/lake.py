import flopy

LAKE_KWDS = ['STATUS', 'STAGE', 'RAINFALL', 'EVAPORATION', 'RUNOFF', 'INFLOW', 'WITHDRAWAL',
             'AUXILIARY', 'RATE', 'INVERT', 'WIDTH', 'SLOPE', 'ROUGH']

def lake_from_gdf(gwf, gdf, ds, recharge=True,
                  claktype='VERTICAL', surfdep=0.05, pname='lak',
                  **kwargs):
    """ add a lake from a geodataframe

    Parameters
    ----------
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    gdf : gpd.GeoDataframe
        geodataframe with the cellids as the index and the columns:
            lakeno : with the number of the lake
            elev : with the bottom of the lake
            strt : with the starting head of the lake
            clake : with the bed resistance of the lake
            optional columns are 'STATUS', 'STAGE', 'RAINFALL', 'EVAPORATION',
            'RUNOFF', 'INFLOW', 'WITHDRAWAL', 'AUXILIARY', 'RATE', 'INVERT', 
            'WIDTH', 'SLOPE', 'ROUGH'. These columns should contain the name
            of a dataarray in ds with the dimension time.
    ds : xr.DataSet
        dataset containing relevant model grid and time information
    recharge : bool, optional
        if True recharge will be added to the lake and removed from the
        recharge package. The recharge 
    claktype : str, optional
        defines the lake-GWF connection type. For now only VERTICAL is
        supported. The default is 'VERTICAL'.
    surfdep : float, optional
        Defines the surface depression depth for VERTICAL lake-GWF connections.
        The default is 0.05.
    pname : str, optional
        name of the lake package. The default is 'lak'.
    **kwargs :
        passed to flopy.mf6.ModflowGwflak.

    Raises
    ------
    NotImplementedError

    Returns
    -------
    lak : flopy lake package

    """
    if claktype != 'VERTICAL':
        raise NotImplementedError('function only tested for claktype=VERTICAL')
        
    if ds.gridtype != 'vertex':
        raise NotImplementedError('only works with a vertex grid')

    assert ds.time.time_units.lower() == 'days', 'expected time unit days'
    time_conversion=86400.0
    # length unit is always meters in nlmod
    length_conversion=1.0
    
    packagedata = []
    connectiondata = []
    perioddata = {}
    for iper in range(ds.dims['time']):
        perioddata[iper] = []
    
    lake_settings = []
    for setting in LAKE_KWDS:
        if setting in gdf.columns:
            lake_settings.append(setting)
            
    lake_settings = [setting for setting in LAKE_KWDS if setting in gdf.columns]

    for lakeno, lake_gdf in gdf.groupby('lakeno'):
        nlakeconn = lake_gdf.shape[0]
        strt = lake_gdf['strt'].iloc[0]
        assert (lake_gdf['strt']==strt).all(), 'a single lake should have single strt'
        packagedata.append([lakeno, strt, nlakeconn])

        iconn = 0
        for icell2d, row in lake_gdf.iterrows():
            cellid = (0, icell2d) # assuming lake in the top layer

            # If BEDLEAK is specified to be NONE, the lake-GWF connection
            # conductance is solely a function of aquifer properties in the
            # connected GWF cell and lakebed sediments are assumed to be absent.
            clake = row['clake']
            bedleak = 1 / clake
            belev = 0.0  # Any value can be specified if CLAKTYPE is VERTICAL
            telev = 0.0  # Any value can be specified if CLAKTYPE is VERTICAL
            connlen = 0.0  # Any value can be specified if CLAKTYPE is VERTICAL
            connwidth = 0.0  # Any value can be specified if CLAKTYPE is VERTICAL
            connectiondata.append([lakeno, iconn, cellid, claktype, bedleak,
                                   belev, telev, connlen, connwidth])
            iconn+=1

        for iper in range(ds.dims['time']):
            if recharge:
                # add recharge to lake
                cellids = [row[2][1] for row in connectiondata]
                rech = ds['recharge'][iper,cellids].values.mean()
                if  rech >= 0:
                    perioddata[iper].append([lakeno, 'RAINFALL', rech])
                    perioddata[iper].append([lakeno, 'EVAPORATION', 0])
                else:
                    perioddata[iper].append([lakeno, 'RAINFALL', 0])
                    perioddata[iper].append([lakeno, 'EVAPORATION', -rech])
                # set recharge to zero
                ds['recharge'][iper,cellids] = 0
            for lake_setting in lake_settings:
                datavar = lake_gdf[lake_setting].iloc[0]
                if not (lake_gdf[lake_setting] == datavar).all():
                    raise ValueError(f'expected single data variable for {lake_setting} and lake number {lakeno}, got {lake_gdf[lake_setting]}')
                
                perioddata[iper].append([lakeno, lake_setting, ds[datavar].values[iper]])
            
                

    lak = flopy.mf6.ModflowGwflak(gwf,
                                  surfdep=surfdep,
                                  time_conversion=time_conversion,
                                  length_conversion=length_conversion,
                                  nlakes=len(packagedata),
                                  packagedata=packagedata,
                                  connectiondata=connectiondata,
                                  perioddata=perioddata,
                                  budget_filerecord=f'{pname}.bgt',
                                  stage_filerecord=f'{pname}.hds',
                                  **kwargs)
    
    return lak