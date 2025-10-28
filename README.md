#GravMag - Python functions for GravMag map inversion

The linear inversion theory is presented in WGC paper by Hokstad et al. (2020). <br />
See the reports and papers in the docs folder: <br />

Test/demo scripts:<br /> 
Synt data tests:<br /> 
    sc_01_ ... : 1 magnet, and varying distance to magnetic source.<br /> 
    sc_02_ ... : 4 magnets, varying the Marquardt-Levenberg regularization.<br /> 
    sc_03_ ... : Deep brick shaped anomaly, 4 shallow anomalies.<br /> 
    sc_04_ ... : 1 magnet. Non-linear Gauss-Newton inversion, updating source-layer depth.<br /> 
    sc_05_ ... : Deep brick shaped anomaly, 4 shallow anomalies. Gauss-Newton.<br /> 
    sc_06_ ... : Two-layer model.<br />

Real data from Iceland:<br /> 
    sc_11_ ... : Aeromag data from Hengill volcanic zone.<br />   
    sc_12_ ... : Aeromag data from Hengill volcanic zone.<br />   

Equivalent-source redatuming:<br /> 
    sc_21_ ... : Prepare data on zyz-file for equivalen-source redatuming<br />   
    sc_22_ ... : Run equivalent source inversion and redatuming<br />   
    sc_23_ ... : Compare some edge-mirroring tests<br />   

Real data magnetic and FTG inversion (Gravity data from URG):<br /> 
    sc_31_ ... : prep for URG inversion<br />   
    sc_32_ ... : Magnetic inversion URG<br />   
    sc_33_ ... : FTG inversion URG<br />   
