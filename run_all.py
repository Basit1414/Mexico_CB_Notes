"""Run the full pipeline end-to-end: data -> EDA -> model selection -> forecast."""
from src import data_pipeline, eda, modeling, forecast

if __name__ == "__main__":
    print("=== 1/4 Data pipeline ===")
    data_pipeline.build_and_save()

    print("\n=== 2/4 EDA ===")
    eda.main()

    print("\n=== 3/4 Model selection (cross-validation) ===")
    modeling.main()

    print("\n=== 4/4 Forecast ===")
    forecast.main()
