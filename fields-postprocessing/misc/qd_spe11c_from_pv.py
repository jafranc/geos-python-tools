import pandas as pd
import numpy as np

boxes = {'Whole': [(0.0, 0.0, 0.0), (8400., 5250, 1350)]}

offx, offz = (100., 150.)
Ox, Oy, Oz = boxes['Whole'][0]
Lx, Ly, Lz = boxes['Whole'][1]
Lx -= Ox
Ly -= Oy
Lz -= Oz

SEC2YEAR = 31.536e6
# schedule = list(np.arange(0, 51 * SEC2YEAR, 5 * SEC2YEAR))
# schedule.extend([75 * SEC2YEAR, 100 * SEC2YEAR])
# schedule.extend(
#     np.arange(150 * SEC2YEAR, 500 * SEC2YEAR, 50 * SEC2YEAR))
# schedule.extend(
#     np.arange(500 * SEC2YEAR, 1001 * SEC2YEAR, 100 * SEC2YEAR))

fd = pd.read_csv('./data/flagged_seal.csv')

boxes['A'] = [(3300., 0., 0.), (8300., 5000., 750.)]
boxes['B'] = [(100., 0., 750.), (3300., 5000., 1350.)]
boxes['C'] = [(Ox + 1.1 / 2.8 * Lx, Oy, Oz + 0.1 / 1.2 * Lz + offz),
              (Ox + 2.6 / 2.8 * Lx, Oy + Ly, Oz + 0.5 / 1.2 * Lz + offz)]


def thread_me(ntask, k):
    sd_time = np.zeros((int(npart / ntask), 1))
    sd_pressure = np.zeros((int(npart / ntask), 2))
    sd_mc = np.zeros((int(npart / ntask), 1))
    sd_mtot = np.zeros((int(npart / ntask), 1))
    sd_masses = {'A': np.zeros((int(npart / ntask), 4)), 'B': np.zeros((int(npart / ntask), 4))}
    for i in range(int(npart / ntask)):
        fname = f'/work/logs/spe11c/t/0/res_41908902/data/spe11c_minimal_0y_{i + k * npart / ntask:0.0f}.csv'
        pos1 = 308592
        pos2 = 270379

        df = pd.read_csv(fname)

        for boxName in ['A', 'B']:
            box = boxes[boxName]
            dfb = df[(df['elementCenter:0'] > box[0][0]) & (df['elementCenter:0'] < box[1][0]) & (
                        df['elementCenter:1'] > box[0][1]) & (df['elementCenter:1'] < box[1][1]) & (
                                 df['elementCenter:1'] > box[0][2]) & (df['elementCenter:1'] < box[1][2])]
            sd_masses[boxName][i, 0] = (dfb[dfb['phaseMobility:0'] < 0.001]['phaseVolumeFraction:0'] *
                                        dfb[dfb['phaseMobility:0'] < 0.001][
                                            'rockPorosity_referencePorosity'] *
                                        dfb[dfb['phaseMobility:0'] < 0.001]['elementVolume'] *
                                        dfb[dfb['phaseMobility:0'] < 0.001]['fluid_phaseDensity:0']).sum()
            sd_masses[boxName][i, 1] = (dfb[dfb['phaseMobility:0'] > 0.001]['phaseVolumeFraction:0'] *
                                        dfb[dfb['phaseMobility:0'] > 0.001][
                                            'rockPorosity_referencePorosity'] *
                                        dfb[dfb['phaseMobility:0'] > 0.001]['elementVolume'] *
                                        dfb[dfb['phaseMobility:0'] > 0.001]['fluid_phaseDensity:0']).sum()
            sd_masses[boxName][i, 2] = (
                    dfb['rockPorosity_referencePorosity'] * dfb['elementVolume'] * dfb[
                'phaseVolumeFraction:1'] * dfb['fluid_phaseDensity:1'] * dfb[
                        'fluid_phaseCompFraction:2']).sum()

            mco2 = dfb['rockPorosity_referencePorosity'] * dfb['elementVolume'] * dfb[
                'phaseVolumeFraction:1'] * dfb[
                       'fluid_phaseDensity:1'] * dfb['fluid_phaseCompFraction:2'] + dfb[
                       'rockPorosity_referencePorosity'] * dfb[
                       'elementVolume'] * dfb['phaseVolumeFraction:0'] * dfb['fluid_phaseDensity:0']

            sd_masses[boxName][i, 3] = (fd['seal_flag'] * mco2).sum()

        sd_pressure[i, :] = np.asarray([df.loc[pos1]['pressure'], df.loc[pos2]['pressure']])
        sd_time[i, 0] = df['Time'][0]

        sd_mtot[i, 0] = (df['rockPorosity_referencePorosity'] * df['elementVolume'] * df['phaseVolumeFraction:1'] * df[
            'fluid_phaseDensity:1'] * df['fluid_phaseCompFraction:2'] + df['rockPorosity_referencePorosity'] * df[
                             'elementVolume'] * df['phaseVolumeFraction:0'] * df['fluid_phaseDensity:0']).sum()

    # if sd_time[i, 0] in schedule:
    #     fname = f'./pt-output/spe11c_spatial_map_{sd_time[i, 0]/SEC2YEAR:0.0f}y.csv'
    #     mco2 = df['rockPorosity_referencePorosity'] * df['elementVolume'] * df['phaseVolumeFraction:1'] * df[
    #         'fluid_phaseDensity:1'] * df['fluid_phaseCompFraction:2'] + df['rockPorosity_referencePorosity'] * df[
    #                'elementVolume'] * df['phaseVolumeFraction:0'] * df['fluid_phaseDensity:0']
    #     dd_df = pd.DataFrame(
    #         data={'x[m]': df['elementCenter:0'], 'y[m]': df['elementCenter:1'], 'z[m]': df['elementCenter:2'],
    #               'pressure[Pa]': df['pressure'], 'gas saturation[-]': df['phaseVolumeFraction:0'],
    #               'mass fraction of CO2 in liquid[-]': df['fluid_phaseCompFraction:2'],
    #               'mass fraction of H2O in vapor[-]': df['fluid_phaseCompFraction:1'],
    #               'phase mass density gas[kg/m3]': df['fluid_phaseDensity:0'],
    #               'phase mass density water [kg/m3]': df['fluid_phaseDensity:1'], 'total mass CO2[kg]': mco2,
    #               'temperature[C]': df['temperature'] - 273.15})
    #     dd_df.to_csv(fname, index=False, float_format='%.3f')

    # after loop
    sd_df = pd.DataFrame(
        data={'t[s]': list(sd_time[:, 0]), 'p1[Pa]': list(sd_pressure[:, 0]), 'p2[Pa]': list(sd_pressure[:, 1]),
              'mobA[kg]': list(sd_masses['A'][:, 0]), 'immA[kg]': list(sd_masses['A'][:, 1]),
              'dissA[kg]': list(sd_masses['A'][:, 2]), 'sealA[kg]': list(sd_masses['A'][:, 3]),
              'mobB[kg]': list(sd_masses['B'][:, 0]), 'immB[kg]': list(sd_masses['B'][:, 1]),
              'dissB[kg]': list(sd_masses['B'][:, 2]), 'sealB[kg]': list(sd_masses['B'][:, 3]),
              'M_C[m]': list(sd_mc[:, 0]), 'sealTot[kg]': list(sd_mtot[:, 0])})

    sd_df.to_csv(f'./pt-output/{k}_spe11c_timeseries.csv', index=False, float_format='%.3f')


npart = 1000
ntask = 500
from tqdm.contrib.concurrent import process_map
from functools import partial

# process_map(partial(thread_me, ntask), range(ntask), max_workers=1)
process_map(partial(thread_me,ntask), range(ntask), max_workers=20)
