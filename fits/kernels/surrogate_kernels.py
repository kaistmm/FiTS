"""
Triton Surrogate Gradient Kernels

This module provides Triton implementations of various surrogate gradient functions
used in Spiking Neural Networks. Each function is designed to match the behavior
of the corresponding PyTorch implementation (see fits/reference.py for types 0-2).

Key Design Principles:
1. Numerical equivalence with PyTorch baseline
2. FP16/BF16 safety (NaN prevention, FP32 promotion where needed)
3. Efficient Triton code generation

Surrogate Type Mapping:
    0: Triangle
    1: Arctangent (atan)
    2: Sigmoid
    3: Exponential  
    4: Rectangle
    5: SuperSpike  (1/(alpha*|x|+1)^2)
"""

import triton
import triton.language as tl
import math

# ============================================================
# Triangle Surrogate Gradient
# ============================================================
@triton.jit
def surrogate_triangle(x, alpha):
    """
    Triangle surrogate gradient function.
    
    PyTorch form:
        grad = (1/gamma)^2 * max(0, gamma - |x|)
    
    Triton Implementation:
        grad = alpha * max(0, 1 - alpha*|x|)
    
    Gamma <-> Alpha Mapping:
        PyTorch gamma = 1 / alpha
        OR equivalently: alpha = 1 / gamma
        
    With this mapping:
        PyTorch: (1/gamma)^2 * max(0, gamma - |x|)
              = alpha^2 * max(0, 1/alpha - |x|)
              = alpha * max(0, 1 - alpha*|x|)  [Triton form]
    
    Args:
        x: Input value (v - threshold)
        alpha: Slope parameter (= 1/gamma in PyTorch)
    
    Returns:
        Gradient value
        
    Properties:
        - Linear decay from threshold
        - Support: |x| < 1/alpha
        - Peak gradient: alpha (at x=0)
    """
    abs_x = tl.abs(x)
    # Scale x by alpha, then check if scaled value < 1
    scaled_abs_x = alpha * abs_x
    # grad = alpha * (1 - scaled_abs_x) for scaled_abs_x < 1, else 0
    result = tl.where(scaled_abs_x < 1.0,
                      alpha * (1.0 - scaled_abs_x),
                      0.0)
    return result


# ============================================================
# Arctangent Surrogate Gradient (Original/Current)
# ============================================================
@triton.jit
def surrogate_atan(x, alpha):
    """
    Arctangent surrogate gradient function.
    
    PyTorch form:
        grad = alpha / (2 * (1 + (pi/2 * alpha * x)^2))
    
    Triton Implementation:
        Same formula
    
    Gamma <-> Alpha Mapping:
        Direct: PyTorch alpha = Triton alpha
        (No transformation needed)
    
    Args:
        x: Input value (v - threshold)
        alpha: Shape parameter
    
    Returns:
        Gradient value
        
    Properties:
        - Smooth decay (no hard cutoff)
        - Infinite support
        - Peak gradient: alpha/2 (at x=0)
    """
    pi_over_2 = 1.5707963267948966
    inner = pi_over_2 * alpha * x
    denom = 1.0 + inner * inner
    return alpha / (2.0 * denom)


# ============================================================
# Sigmoid Surrogate Gradient
# ============================================================
@triton.jit
def surrogate_sigmoid(x, alpha):
    """
    Sigmoid surrogate gradient function.
    
    PyTorch form:
        sgax = sigmoid(alpha * x)
        grad = alpha * sgax * (1 - sgax)
    
    Triton Implementation:
        Same, with FP32 promotion and clamping for stability
    
    Gamma <-> Alpha Mapping:
        Direct: PyTorch alpha = Triton alpha
    
    Args:
        x: Input value (v - threshold)
        alpha: Steepness parameter
    
    Returns:
        Gradient value
        
    Properties:
        - Smooth S-shaped curve
        - Infinite support
        - Peak gradient: alpha/4 (at x=0)
        
    Safety:
        - Clamp input to prevent exp overflow
        - Use FP32 for exp computation
    """
    # Clamp input to prevent overflow in exp
    # exp(-88) ≈ 3e-39 (safe), exp(88) ≈ 3e38 (safe in FP32)
    x_clamped = tl.where(x > 88.0, 88.0,
                         tl.where(x < -88.0, -88.0, x))
    
    # Compute sigmoid in FP32 for stability
    ax = alpha * x_clamped
    # sigmoid(x) = 1 / (1 + exp(-x))
    exp_neg_ax = tl.exp(-ax)
    sigmoid_val = 1.0 / (1.0 + exp_neg_ax)
    
    # grad = alpha * sigmoid * (1 - sigmoid)
    result = alpha * sigmoid_val * (1.0 - sigmoid_val)
    return result


# ============================================================
# Exponential Surrogate Gradient
# ============================================================
@triton.jit
def surrogate_exp(x, alpha):
    """
    Exponential surrogate gradient function.
    
    PyTorch form:
        grad = (alpha / 2) * exp(-alpha * |x|)
    
    Triton Implementation:
        Same, with clamping for stability
    
    Gamma <-> Alpha Mapping:
        Direct: PyTorch alpha = Triton alpha
    
    Args:
        x: Input value (v - threshold)
        alpha: Decay rate parameter
    
    Returns:
        Gradient value
        
    Properties:
        - Exponential decay from threshold
        - Infinite support
        - Peak gradient: alpha/2 (at x=0)
        
    Safety:
        - Clamp exp argument to prevent underflow
    """
    abs_x = tl.abs(x)
    # Clamp to prevent underflow: exp(-88) ≈ 3e-39
    exp_arg = -alpha * abs_x
    exp_arg_clamped = tl.where(exp_arg < -88.0, -88.0, exp_arg)
    
    result = (alpha / 2.0) * tl.exp(exp_arg_clamped)
    return result


# ============================================================
# Rectangle Surrogate Gradient
# ============================================================
@triton.jit
def surrogate_rectangle(x, alpha):
    """
    Rectangle (box) surrogate gradient function.
    
    PyTorch form:
        grad = 1.0 if |x| < 0.5 else 0.0
    
    Triton Implementation:
        Same (alpha parameter ignored for compatibility)
    
    Gamma <-> Alpha Mapping:
        N/A (no parameter used)
    
    Args:
        x: Input value (v - threshold)
        alpha: Unused (kept for API consistency)
    
    Returns:
        Gradient value
        
    Properties:
        - Constant gradient in window
        - Hard cutoff at boundaries
        - Support: |x| < 0.5
    """
    abs_x = tl.abs(x)
    result = tl.where(abs_x < 0.5, 1.0, 0.0)
    return result


# ============================================================
# SuperSpike Surrogate Gradient
# ============================================================
@triton.jit
def surrogate_superspike(x, alpha):
    """
    SuperSpike surrogate gradient: 1 / (alpha*|x| + 1)^2

    This matches the test reference's s_type=0 (PyTorch side).
    Triton surrogate_type=5.
    """
    abs_x = tl.abs(x)
    denom = alpha * abs_x + 1.0
    return 1.0 / (denom * denom)


# ============================================================
# Surrogate Selection Helper (for compile-time dispatch)
# ============================================================
@triton.jit
def select_surrogate_grad(x, alpha, surrogate_type: tl.constexpr):
    """
    Select and compute surrogate gradient based on type.

    This is a compile-time dispatch function. The surrogate_type
    must be a constexpr (compile-time constant), allowing Triton
    to optimize away unused branches.

    Args:
        x: Input value
        alpha: Surrogate parameter
        surrogate_type: Surrogate type ID (constexpr)
            0: Triangle
            1: Arctangent
            2: Sigmoid
            3: Exponential
            4: Rectangle
            5: SuperSpike (1/(alpha*|x|+1)^2)

    Returns:
        Gradient value from selected surrogate
    """
    if surrogate_type == 0:
        return surrogate_triangle(x, alpha)
    elif surrogate_type == 1:
        return surrogate_atan(x, alpha)
    elif surrogate_type == 2:
        return surrogate_sigmoid(x, alpha)
    elif surrogate_type == 3:
        return surrogate_exp(x, alpha)
    elif surrogate_type == 4:
        return surrogate_rectangle(x, alpha)
    elif surrogate_type == 5:
        return surrogate_superspike(x, alpha)
    else:
        # Fallback to atan (original default)
        return surrogate_atan(x, alpha)
