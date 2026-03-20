from src.training_code import TrainerClient

if __name__ == "__main__":
    trainer = TrainerClient(model_name="forecasting_usecase")
    trainer.run()