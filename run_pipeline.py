import click
from pipeline.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
def main():
    run = ml_pipeline()
if __name__ == "__main__":
    main()
