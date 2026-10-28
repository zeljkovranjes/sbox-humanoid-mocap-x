namespace HumanoidMocapX.TensorInference;

/// <summary>Elementwise activation used by transformer feed-forward layers.</summary>
public enum TensorActivation
{
    /// <summary>x multiplied by its logistic sigmoid.</summary>
    Silu,
    /// <summary>GELU using the tanh approximation.</summary>
    Gelu,
    /// <summary>GELU using the error-function definition with a polynomial erfc approximation.</summary>
    GeluErf
}
