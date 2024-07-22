# Diseased Vasculatures and Microbubble Dynamics

## Overview

Understanding the dynamics of micro and nanoparticles in disturbed flow profiles of diseased vasculatures is critical for advancing targeted treatments and developing effective therapeutic strategies. Our research explores the use of a novel ultrasound method to fill aneurysm cavities with clinically approved microbubbles (MBs). Using microfluidics-based aneurysm models that mimic physiological flow conditions, we have developed a groundbreaking technique for trapping and aggregating MBs in disturbed flow environments.

## Key Findings

### Microbubble Trapping and Aggregation

When MBs are injected into disturbed flow and ultrasound is activated, they become trapped at the vortex's eye. They attract other MBs from the flow due to ultrasound-induced attractive forces.

<video width="600" controls>
  <source src="videos/Bubblu_vortex_20fps.avi" type="video/avi">
  Your browser does not support the video tag.
</video>

### Cluster Formation and Ejection

Once the MB cluster reaches a critical size, it is ejected from the vortex and migrates to the wall opposite the piezo transducer, where it continues to attract and cluster MBs, filling the aneurysm cavity.

### Therapeutic Potential

This technique has the potential to fill aneurysm cavities within minutes and may be crucial in treating vascular diseases associated with plaque deposits.

## One-Sentence Summary

Novel acoustic-based self-assembly technique using microbubbles explores disturbed flow effects under ultrasound, promising targeted drug delivery in diseased vasculatures.

## Experimental and Theoretical Framework

### Governing Equations

The behavior of MBs under disturbed flow and ultrasound is governed by the following equations:

#### Microbubble Position and Velocity

$$
\frac{dx}{dt} = u_{MB}
$$

$$
\frac{du_{MB}}{dt} = \frac{3}{\rho} \nabla p + \frac{3}{4} C_D (u - u_{MB}) |u - u_{MB}|
$$

where \( C_D \) is the drag coefficient, \( p \) is the pressure field, \( x \) is the microbubble position, and \( u_{MB} \) is its velocity.

#### Vortex Velocity Field

$$
u = 
\begin{cases} 
-u_{\theta} \sin \theta, u_{\theta} \cos \theta \\
u_{\theta} = \frac{\Gamma}{2 \pi} \frac{r}{a^2}, & \text{when } r < a \\
u_{\theta} = \frac{\Gamma}{2 \pi} \frac{1}{r}, & \text{when } r > a 
\end{cases}
$$

with \( \sin \theta = \frac{y}{r} \), \( \cos \theta = \frac{x}{r} \), and \( r = \sqrt{x^2 + y^2} \), and \( a \) is the radius of the vortex core.

#### Pressure Field

$$
p = 
\begin{cases} 
p_{\infty} - \frac{\Gamma^2}{4 \pi^2} \frac{\rho}{a^2} + \frac{\rho \Gamma^2}{8 \pi^2} \frac{r^2}{a^2}, & \text{when } r < a \\
p_{\infty} - \frac{\Gamma^2}{8 \pi^2} \frac{\rho}{r^2}, & \text{when } r > a 
\end{cases}
$$

### Combined Equation

Combining the first two equations results in a second-order partial differential equation:

$$
\frac{d^2 x}{dt^2} + a \frac{dx}{dt} + bx = cu
$$

This equation can be solved to determine the bubble position over time if the initial coordinates are known.

## Implementation and Visualization

### Python Scripts

Our GitHub repository includes Python scripts that:
- Track experimental videos and plot the data.
- Solve the theoretical governing equations.
- Plot the trajectory of microbubbles using velocity data extracted from COMSOL simulations.

### Visualization

The plots illustrate the trajectories of single and multiple microbubbles in disturbed flow environments, providing insights into their behavior under ultrasound.

## Conclusion

Our research presents a novel method for using ultrasound to manipulate microbubbles in disturbed flow environments, offering promising applications for targeted drug delivery in diseased vasculatures. This technique holds potential for significant advancements in vascular disease treatments, particularly for conditions involving aneurysms and plaque deposits.

## Repository Contents

- **Scripts:** Python scripts for solving theoretical equations and plotting microbubble trajectories.
- **Data:** Experimental videos and velocity data from COMSOL simulations.
- **Documentation:** Detailed documentation on how to run the scripts and interpret the results.
