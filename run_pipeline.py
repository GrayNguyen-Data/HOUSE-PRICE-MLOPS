import click
from pipeline.training_pipeline import ml_pipeline


@click.command()
def main():
    run = ml_pipeline()
    # Execute the pipeline run so ZenML can evaluate caching
    run.run()
if __name__ == "__main__":
    main()
