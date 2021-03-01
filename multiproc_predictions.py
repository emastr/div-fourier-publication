from error_estimation.windfield_prediction import do_parallel_5fold
from pathos.multiprocessing import freeze_support
import framework.data_loader as dl

# Import windfields
from models.kriging_windfield import KrigingWindfield
from models.random_forest_poolcheat import RandomForestField
from models.mlp_windfield_poolcheat import MLPWindfield
from models.quadratic_fourier_windfield import FourierWindfield
from models.weighting_windfield import WeightingWindfield
from models.averaging_windfield import AveragingWindfield
from models.residual_windfield import ResidualWindfield
from models.nn_windfield import NearestNeighbourWindfield
from models.MCMC_fourier_windfield import RandomFourierFeatures


data_folder = "D:/python_runs/windmodelling/Data_nostorm/"
save_folder = "D:/python_runs/windmodelling/Runs_diverse_new_seed/"

if __name__ == "__main__" and __package__ is None:
    freeze_support()
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))

    data = dl.load_parquet("https://www.tangel.se/wind/old/wind_frames/wind-2018.parquet")
    data = data[data['date'].apply(lambda f: f[0:7]) != '2018-09']
    # split_and_save_data(data, data_folder)
    # data = date_time_to_datetime(data)
    lookup_data = data[['x', 'y', 'altitude']].drop_duplicates()

    # Define windfields
    mlp = MLPWindfield(epochs=900, learning_rate=0.001, layers=[20, 20, 20, 2], elevation=True, l2 = 0.01, gamma=0.01, N=0, altitude_data=lookup_data)# 900 epochs
    nf = NearestNeighbourWindfield()
    rf = RandomForestField(200, lookup_data, 3)
    af = AveragingWindfield(exponent=3)
    fs = FourierWindfield((21,21), 'sobolev', 0.017, 0.047)
    mc = RandomFourierFeatures(n_terms=441, n_steps=500, reg_param=0.017, div_param=0.001, gamma=1.25, sigma=2.25, seed=100)
    wf = WeightingWindfield([rf, fs], [0.48, 0.52])
    kf = KrigingWindfield(variogram_model="exponential")
    rs = ResidualWindfield([fs, kf])

    # put in a list
    wfs = [mlp, nf, rf, af, fs, mc, wf, kf, rs]

    # Make predictions
    pr = do_parallel_5fold(windfields=wfs, data_folder=data_folder, save_folder=save_folder, processes=6, random=True, n_samples=500, seed=0)