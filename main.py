from hydro_inflow.hydro_config import get_config
import reproduce.prepare_and_check_hydro_inputs as prepare
import argparse
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

#Only here you will need the args, and the args will help you to define the processes to activate in the entire code
def parse_args() -> argparse.Namespace:
    #... your code for parsing arguments
    pass # to erase when you implement the function


if __name__ == "__main__":
    args = parse_args()

    #from prepare_hydro_inputs
    cfg = get_config() # you were passing repo_root, but i am not sure that it is used

    logger.debug("Config loaded: %s", cfg)

    # if you run everything correctly from one script like this main, you do not need to "ensure directories". Plus, you do not need them empty at the beginning, you can create them step by step.

    # Download here all the inputs with one method, e.g. prepare.download_inputs(args), and inside that method you can decide which inputs to download based on the args. You can also create the directories if they do not exist, but you do not need to ensure that they are empty at the beginning, because you will be filling them with the downloaded data.
    # Plus, why do you have skip-static, skip-glofas, etc. as arguments? When don't you need to download them?

    prepare.prepare_hydro_plants() # For ANY reason you have to "run a script as a thread", you can just call the function directly, you do not need to run it as a subprocess and passing ONLY the necessary args. See how i am calling prepare_hydro_inputs.py

    prepare.build_entsoe_hydro_annual_production() # For ANY reason you have to "run a script as a thread", you can just call the function directly, you do not need to run it as a subprocess and passing ONLY the necessary args. See how i am calling prepare_hydro_inputs.py

    # The following function are useful to check inputs. Inputs need to be checked in the download part. If there is an error, is there that you need to RAISE an error
    #    log_input_status(data_root=data_root, repo_root=repo_root, cfg=cfg)
    #log_manual_input_instructions(data_root=data_root, cfg=cfg)



# Example of good implementation of prepare_hydro_plants() that you should have in prepare_hydro_inputs.py - i haven't tested it on the whole environment
# Of course all the import statements should be at the top of the file.
import utils

def prepare_hydro_plants(log_level=logging.INFO) -> None:
    # From what i see input and output are defined and they do not change, so you do not need args
    # otherwise input_path and output_path are parameters -- NOTE: repo_root needs to disapper from ANYWHERE
    utils.setup_logging(log_level) #note that you are not passing the default level, you should. It is a parameter usually written in configs and you need to continously pass it.

    if not input_path.exists():
        raise FileNotFoundError(f"Input GloHydroRes file not found: {input_path}")

    logger.info("Reading GloHydroRes source: %s", input_path)

    df = load_and_prepare_glohydrores(input_path=input_path)
    logger.info("Loaded and converted plants: %s", len(df))

    df = convert_glohydrores_country_to_iso2(df)
    df = filter_to_pypsa_countries(df, countries=PYPSA_EUR_COUNTRIES)
    df = apply_manual_glohydrores_corrections(df)

    df = fill_hydro_nan_parameters(
        glohydro_df=df,
        verbose=args.verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy() # Not necessary
    df.index = df["source_id"].astype(str)
    df.index.name = "plant_index"
    df.to_csv(output_path, index=True)

    logger.info("Prepared hydropower plant file saved: %s", output_path)
    logger.info("Rows saved: %s", len(df))
    logger.info("Columns saved: %s", len(df.columns))

    # From here you can call the methods in run_hydro_inflow_framework


#NOTE: If you are using a method multiple times and this is quite generic, like setup_logging this should be in a file call utils.py and you should call it from it, not writing it again and again
# repo_root/data_root should disappear, eventually data_root can work, but not as environment variable as you implemented

# If you correctly implement the main above you don't need to "check_hydro_inputs"
# since you will have all the necessary for proceeding with the modeling
# Each method should take care of checking the inputs that it needs, and if there is an error, it should raise an error, not just log it. As a consequence, you don't need check_hydro_inputs.py

# Another important thing: there is ONE AND ONLY ONE main that you activate

# All the framework needs to be activated from THIS main
# Then, there will be another command to run PyPSA with all the modification
# NOTE: PyPSA implementation is definitely NOT a good example of good object-oriented implementation in CS
# Since our model can be also applied to other energy systems, we need to correctly differentiate between OUR framework and what has been integrated with PyPSA.