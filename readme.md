Contains FEA surrogate code, trained model and sample data for sub sea pressure vessel design

The pressure vessel is a critical component in modern-day Unmanned Underwater vehicle, Remotely operated Vehicles, etc. It is designed to withstand the sub-sea hydro-static pressure conditions while remaining watertight and contains power sources, electronic and other sensors that cannot be flooded. For this purpose, FEA-based simulation is conducted to measure the maximum induced stress in a design subject to the subsea pressurized environment. The simulation consists of multiple steps - CAD modeling, body meshing, and numerical solution of FEA to get the stress distribution.   
 The measured maximum static stress is compared with the yield strength of the material to check the integrity of the pressure vessel during operation.  
 
 
The motivation for learning a surrogate is the ability to interpolate or predict the numerical simulators' output at a very low cost using trainable models.

![Alt text](https://github.com/vardhah/FEA_surrogate/blob/main/hull_surrogate.png)


The material used to design the vessel is Aluminium alloy (Al6061-T6) which is the most commonly used material for pressure vessel design.   
    
Our NN architecture is a multilayered NN with 9 layers. Our network has four major constituents- fully connected layer (linear+ReLU), dropout layer, skip connections, and output layer. The input layer is the concatenation of the normalized vector of design variables and sub-sea pressure ($[D_{sea}, L_V, Th_V, R_{end}]$)
   
