import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def solve_circuit():
    # --- 1. CONFIGURATION ---
    # Frequencies where you want the current to SPIKE (Impedance = 0)
    # STRATEGY: To widen the 50Hz band, we use "Stagger Tuning".
    # We target 45Hz and 55Hz instead of just 50Hz. This creates a wider "valley".
    target_freqs_hz = [40, 60, 100, 250] 
    
    # Component Values (Fixed)
    L1 = 4.7e-3   # 6.7 mH (Load)
    L2 = 4.7e-3   # Tank 1
    L3 = 4.7e-3   # Tank 2
    L4 = 4.7e-3   # Tank 3 (Added for stagger tuning)

    # Convert targets to angular frequency (rad/s)
    w_targets = np.array([2 * np.pi * f for f in target_freqs_hz])

    # --- 2. THE PHYSICS ---
    # Reactance X(w) = wL1 - 1/wC1 + wL2/(1 - w^2L2C2) + ...
    def calculate_reactance(w, C1, C2, C3, C4):
        # Term 1: Series L1 and C1
        # Handle w=0 case to avoid division by zero
        if w == 0: return -np.inf 
        
        X_series = (w * L1) - (1.0 / (w * C1))
        
        # Term 2: Parallel Tank 1 (L2 || C2)
        denom2 = 1.0 - (w**2 * L2 * C2)
        if abs(denom2) < 1e-12: denom2 = 1e-12
        X_tank1 = (w * L2) / denom2
        
        # Term 3: Parallel Tank 2 (L3 || C3)
        denom3 = 1.0 - (w**2 * L3 * C3)
        if abs(denom3) < 1e-12: denom3 = 1e-12
        X_tank2 = (w * L3) / denom3

        # Term 4: Parallel Tank 3 (L4 || C4)
        denom4 = 1.0 - (w**2 * L4 * C4)
        if abs(denom4) < 1e-12: denom4 = 1e-12
        X_tank3 = (w * L4) / denom4
        
        return X_series + X_tank1 + X_tank2 + X_tank3

    def reactance_residuals(vars):
        C1, C2, C3, C4 = vars
        residuals = []
        for w in w_targets:
            res = calculate_reactance(w, C1, C2, C3, C4)
            residuals.append(res)
        return residuals

    # --- 3. INITIAL GUESSES ---
    c1_guess = 1 / (w_targets[0]**2 * L1)
    c2_guess = 1 / (w_targets[1]**2 * L2) 
    c3_guess = 1 / (w_targets[2]**2 * L3)
    c4_guess = 1 / (w_targets[3]**2 * L4)

    initial_guess = [c1_guess, c2_guess, c3_guess, c4_guess]

    print(f"Targeting Frequencies: {target_freqs_hz} Hz")
    print("Solving (Staggering 45/55Hz for width)...")

    # --- 4. SOLVE ---
    result = least_squares(reactance_residuals, initial_guess, bounds=(0, np.inf), max_nfev=5000)

    if result.success:
        C1_sol, C2_sol, C3_sol, C4_sol = result.x
        
        print("\nSUCCESS! Found component values:")
        print("-" * 40)
        print(f"L1 (Load):   {L1*1000} mH")
        print(f"C1 (Series): {C1_sol*1e6:.4f} uF")
        print("-" * 40)
        print(f"L2 (Tank 1): {L2*1000} mH")
        print(f"C2 (Tank 1): {C2_sol*1e6:.4f} uF")
        print("-" * 40)
        print(f"L3 (Tank 2): {L3*1000} mH")
        print(f"C3 (Tank 2): {C3_sol*1e6:.4f} uF")
        print("-" * 40)
        print(f"L4 (Tank 3): {L4*1000} mH")
        print(f"C4 (Tank 3): {C4_sol*1e6:.4f} uF")
        print("-" * 40)
        
        # --- 5. BODE PLOT ---
        print("\nGenerating Bode Plot...")
        
        # Create a frequency range for plotting
        freqs_plot = np.logspace(0, np.log10(max(target_freqs_hz)*10), 1000)
        w_plot = 2 * np.pi * freqs_plot
        
        Z_mag = []
        R_parasitic = 0.1 # Add 0.1 Ohm parasitic resistance
        
        for w in w_plot:
            X_total = calculate_reactance(w, C1_sol, C2_sol, C3_sol, C4_sol)
            Z = np.sqrt(R_parasitic**2 + X_total**2)
            Z_mag.append(Z)
            
        plt.figure(figsize=(10, 6))
        plt.loglog(freqs_plot, Z_mag, 'b-', linewidth=1.5, label='Impedance |Z|')
        
        # Plot markers at target frequencies
        for f in target_freqs_hz:
            plt.axvline(x=f, color='r', linestyle='--', alpha=0.6)
            plt.text(f, plt.ylim()[0] * 1.5, f'{f} Hz', ha='center', va='bottom', fontsize=9, color='r')
            
        # Highlight the widened region
        # plt.axvspan(50, 60, color='yellow', alpha=0.2, label='Widened 50Hz Region')

        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Impedance (Ohms)')
        plt.title('Multi-Resonant Circuit (Stagger Tuned)')
        plt.legend()
        plt.ylim(0.01, 1000) 
        plt.show()

    else:
        print("Solver failed to converge.")

if __name__ == "__main__":
    solve_circuit()