namespace RealEstatePricePredictionModel
{
    /// <summary>Hyperparameters and options of one training run.</summary>
    public class TrainingConfig
    {
        public int HiddenNodes = 16;
        public double LearningRate = 0.3;     // applied to both layers
        public int Epochs = 500;
        public int Seed = 1;                  // weight initialisation + sample shuffling
        public int[] CheckpointEpochs = new int[0]; // epochs at which train/val metrics are recorded (tuning)
        public bool EvaluateOnTest = false;   // only the final run touches the test set
    }
}
