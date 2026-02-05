import numpy as np
from scipy.optimize import least_squares

def solve_circuit():
    # --- 1. CONFIGURATION ---
    # Frequencies where you want the current to SPIKE (Impedance = 0)
    # We now have 4 distinct frequencies.
    target_freqs_hz = [10, 50, 100, 250] 
    
    # Component Values (Fixed)
    # Added L4 for the new stage.
    L1 = 6.7e-3  # 6.7 mH (Load)
    L2 = 80e-3  # 9.6 mH (Tank 1)
    L3 = 80e-3  # 9.6 mH (Tank 2)
    L4 = 80e-3  # 9.6 mH (Tank 3 - New)

    # Convert targets to angular frequency (rad/s)
    w_targets = np.array([2 * np.pi * f for f in target_freqs_hz])

    # --- 2. THE PHYSICS ---
    # Reactance X(w) = X_series + X_tank1 + X_tank2 + X_tank3
    def reactance_residuals(vars):
        # Unpack 4 variables now
        C1, C2, C3, C4 = vars
        
        residuals = []
        for w in w_targets:
            # Term 1: Series L1 and C1
            X_series = (w * L1) - (1.0 / (w * C1))
            
            # Term 2: Parallel Tank 1 (L2 || C2)
            denom2 = 1.0 - (w**2 * L2 * C2)
            if abs(denom2) < 1e-9: denom2 = 1e-9 
            X_tank1 = (w * L2) / denom2
            
            # Term 3: Parallel Tank 2 (L3 || C3)
            denom3 = 1.0 - (w**2 * L3 * C3)
            if abs(denom3) < 1e-9: denom3 = 1e-9
            X_tank2 = (w * L3) / denom3
            
            # Term 4: Parallel Tank 3 (L4 || C4) - NEW
            denom4 = 1.0 - (w**2 * L4 * C4)
            if abs(denom4) < 1e-9: denom4 = 1e-9
            X_tank3 = (w * L4) / denom4
            
            # Total Reactance should be 0
            residuals.append(X_series + X_tank1 + X_tank2 + X_tank3)
            
        return residuals

    # --- 3. INITIAL GUESSES ---
    # We need a guess for C4 now as well.
    # C1 usually resonates with L1 at the lowest freq.
    c1_guess = 1 / (w_targets[0]**2 * L1)
    
    # Tanks usually resonate somewhat higher than the base freq
    c2_guess = 1 / (w_targets[1]**2 * L2) 
    c3_guess = 1 / (w_targets[2]**2 * L3)
    c4_guess = 1 / (w_targets[3]**2 * L4)

    initial_guess = [c1_guess, c2_guess, c3_guess, c4_guess]

    print(f"Targeting Frequencies: {target_freqs_hz} Hz")
    print("Solving for 4 stages...")

    # --- 4. SOLVE ---
    # bounds=(0, np.inf) ensures we don't get negative capacitors
    result = least_squares(reactance_residuals, initial_guess, bounds=(0, np.inf), max_nfev=5000)

    if result.success:
        C1_sol, C2_sol, C3_sol, C4_sol = result.x
        
        print("\nSUCCESS! Found component values:")
        print("-" * 40)
        print(f"{target_freqs_hz[0]} Hz")
        print(f"L1 (Load):   {L1*1000} mH")
        print(f"C1 (Series): {C1_sol*1e6:.4f} uF")
        print("-" * 40)
        print(f"{target_freqs_hz[1]} Hz")
        print(f"L2 (Tank 1): {L2*1000} mH")
        print(f"C2 (Tank 1): {C2_sol*1e6:.4f} uF")
        print("-" * 40)
        print(f"{target_freqs_hz[2]} Hz")
        print(f"L3 (Tank 2): {L3*1000} mH")
        print(f"C3 (Tank 2): {C3_sol*1e6:.4f} uF")
        print("-" * 40)
        print(f"{target_freqs_hz[3]} Hz")
        print(f"L4 (Tank 3): {L4*1000} mH")
        print(f"C4 (Tank 3): {C4_sol*1e6:.4f} uF")
        print("-" * 40)
        
        # Verification check
        print("\nVerification (Reactance at targets should be ~0 Ohms):")
        final_errors = reactance_residuals(result.x)
        for f, err in zip(target_freqs_hz, final_errors):
            print(f"Freq {f} Hz -> Error: {err:.4f} Ohms")
    else:
        print("Solver failed to converge.")

if __name__ == "__main__":
    solve_circuit()