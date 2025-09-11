# Cold Atom MOT Simulation

This project simulates the dynamics of cold Rubidium atoms in a Magneto-Optical Trap (MOT) using a **Verlet integration scheme**.  
The code initializes atomic positions and velocities, then evolves them in time under a one-body optical potential.

## Requirements

- Python 3.9+
- Dependencies:
  - `numpy`
  - `matplotlib`
  - `pandas`

Install them via:

```bash
pip install numpy matplotlib pandas
````

---

## How to Run

1. Clone the repository.
2. Run the simulation script (example):

```bash
python simulation.py
```

3. The script will:

   * Initialize atoms with random positions and velocities.
   * Integrate their trajectories using the Verlet algorithm.
   * Plot sample trajectories.

---

## Simulation Parameters

* **Temperature**

  * `T = 15 × 10⁻⁶ K`

* **MOT Geometry**

  * `d_mot = 5 mm` (MOT displacement)
  * `R_mot = 1 mm` (MOT radius)
  * `MOT_t = 0.1 s` (MOT duration)
  * `z_max = R_mot + d_mot` (maximum axial extent)
  * `rho_max = z_max / zR` (maximum radial extent, scaled by Rayleigh length `zR`)

---

## Atoms

* **Number of atoms**

  * `N = 1 × 10²` (100 atoms)

* **Initial positions**

  * Radial coordinate:
    `rho_0 ∼ U(0, rho_max)`
  * Axial coordinate:
    `zeta_0 ∼ U(0, rho_max)`
  * Combined:
    `x0 = [rho_0, zeta_0]`

* **Velocity scales**

  * Radial scaling: `vs_rho = w0 / tau`
  * Axial scaling: `vs_zeta = zR / tau`
  * Thermal factor: `alpha = m_Rb / (2 kB T)`
  * Thermal velocity scale: `v_bar = sqrt(π / alpha)`

* **Initial velocities**

  * Radial:
    `v_rho_0 ∼ N(0, 1/(2α)) × vs_rho / v_bar` NEEDS CORRECTION
  * Axial:
    `v_zeta_0 ∼ N(0, 1/(2α)) × vs_zeta / v_bar`
  * Combined:
    `v0 = [v_rho_0, v_zeta_0]`

---

## Time Discretization

* Total simulation time (normalized):
  `t_max = MOT_t / tau`

* Time step:
  `time_step = t_max / 1 × 10³`

* Number of steps:
  `N_steps = int(t_max / time_step)`

---

## Dynamics Model

### Physical constants

* Speed of light: `c = 299792458 m/s`
* Boltzmann constant: `kB = 1.38064852 × 10⁻²³ J/K`
* Gravity: `g = 9.81 m/s²`
* Rubidium-87 mass: `m_Rb = 87 × 1.66054 × 10⁻²⁷ kg`

### Trap setup

* Laser wavelength: `λ = 1064 nm`
* Beam power: `P_b = 2 W`
* Fiber radius: `R_trap = 30 μm`
* Beam waist: `w0 = 19 μm`
* Rayleigh length: `zR = π w0² / λ`
* Intensity scale: `I0 = 2 P_b / (π w0²)`
* Optical potential scale: `φ = k I0 / (m_Rb w0²)`
* Time scale: `τ = 1 / sqrt(φ)`

### Force & acceleration

The trapping potential is modeled via functions of scaled coordinates `(ρ, ζ)`:

* Envelope:
  `β(ζ) = 1 / (1 + ζ²)`

* Potential derivatives:

  * `du/drho = -4 β(ζ)² ρ exp(-2 β(ζ) ρ²)`
  * `du/dzeta = -4 β(ζ) ζ exp(-2 β(ζ) ρ²) (1 - 2 β(ζ) ρ²)`

* Acceleration (2D vector):

  ```python
  def acc(rho, zeta):
      acc_rho = du_drho(rho, zeta)
      acc_zeta = du_dzeta(rho, zeta)
      return np.array([acc_rho, acc_zeta])
  ```

---

## Output

* Time evolution of atom positions and velocities.
* Visualizations (2D or 3D plots with `matplotlib`).
* Extendable to compute density profiles, temperature evolution, or escape rates.

---

## Notes

* Units are mixed (meters, seconds, normalized units). Be consistent in analysis.
* The current model uses only the **optical dipole trap potential** (one-body).
* Future extensions: include atom-atom interactions or external fields.


## 🤝 Collaboration Guidelines

This repository is private. If you have been added as a collaborator, you can clone the repository directly:

```bash
git clone https://github.com/Leonardo-Razzai/MD_simulation
.git
cd MD_simulation
```

### Running Simulations

1. Ensure you have Python 3.9+ and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run a single simulation with specified MOT temperature `T` (in μK) and displacement `dMOT` (in mm):

   ```bash
   python simulation.py 15 8
   ```

   This will simulate atoms at **T = 15 μK** with a MOT displaced **8 mm** from the fiber tip.
3. Run batch simulations across a range of parameters:

   ```bash
   python run_multiple_simul.py
   ```

   Results will be stored in the `data/` folder and plots in `img/`.

### Contributing Changes

* Always create a **new branch** before making changes:

  ```bash
  git checkout -b feature-description
  ```
* Push your branch to the repository:

  ```bash
  git push origin feature-description
  ```
* Open a **Pull Request (PR)** from your branch into `master`.
  In your PR, briefly describe the changes and any new physics or analysis methods added.
* All code should follow PEP8 style and include **docstrings** for new functions.

  ## To Do
  ### Simulation wise:
  1) Add heating due to temperature
  2) Extract the actual fraction of atoms trapped from the MOT
  3) Extend to the atoms in the fiber to estimate their lifeltime in the fiber
  4) Add effect of collisions with background gas
  5) Use the real gaussian beam
 
  ### Anlaysis
  1) Plot radial distribution at the fiber as a function of time
  2) Extract velocity distribution
