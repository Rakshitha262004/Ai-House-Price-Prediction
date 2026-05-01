# ============================================
# ENTRY POINT
# HOUSE PRICE PREDICTION
# ============================================

from src.train import train_model

if __name__ == "__main__":
    print("=" * 50)
    print("  HOUSE PRICE PREDICTION — TRAINING PIPELINE")
    print("=" * 50)
    model = train_model()
    print("\nPipeline complete. Model is ready for inference.")