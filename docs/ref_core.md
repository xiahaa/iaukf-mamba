# Augmented State Estimation of Line Parameters in Active Power Distribution Systems With Phasor Measurement Units

Yubin Wang, Mingchao Xia  $①$ , Senior Member, IEEE, Qiang Yang  $①$ , Senior Member, IEEE, Yuguang Song, Graduate Student Member, IEEE, Qifang Chen  $①$ , Member, IEEE, and Yuanyi Chen  $①$



![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-28/7f406345-5a93-423a-8be2-3cd72cf2cd67/ef0055ff3ad27745d7bc387488d40d05757f43b5f970df499decdba9f89ac79b.jpg)



# II. OVERVIEW OF THE PROPOSED SOLUTION

The framework of the method proposed in this study is illustrated in Fig. 1. The DMS can run the application of SE after receiving the measurement data from the SCADA and the PMU. Then the suspicious line parameters can be determined through the analysis of the measurement normalized residuals calculated based on the results of SE. The accurate values of the suspicious line parameters are estimated by the method we proposed, as shown within the dotted box in Fig. 1. Considering the time-invariant characteristics of the parameters, the state equation of the suspicious line parameters is established firstly. And the measurement equation is modified to the form that takes into account the suspicious parameters. Then the distribution network augmented state-space model is established by combining the state equation of the system state variables (voltage amplitude and phase angle of each bus), the parameter state equation and the modified measurement equation. Observing that the line parameters estimation is off-line, the augmented state-space model can be extended to the form under multiple

measurement snapshots. The IAUKF consists of a UKF and a new fault-tolerance NSE [36]. The details of IAUKF are described in Section III. The NSE of IAUKF can accurately perceive the process noise statistical parameters and ensure the robustness of the algorithm during operation. Thus, the values of the suspicious line parameters can be estimated accurately by the augmented state estimation based on the IAUKF because the process noise statistical parameters of the system state variables and suspicious line parameters can be perceived by the NSE in the IAUKF.

# III. THE IMPROVED ADAPTIVE UKF

The UKF is a nonlinear filter that is widely used in the field of dynamic state estimation of power systems [31], [37]–[39]. But it can only guarantee superior performance if the process noise variance matrix  $\mathbf{Q}_k$  and the measurement noise variance matrix  $\mathbf{R}_k$  are accurately known. In the line parameters estimation problem,  $\mathbf{R}_k$  can be obtained from the nameplate data of the measurement equipment and can be considered as known. However,  $\mathbf{Q}_k$  is unknown and time-varying because the suspicious parameters are unknown and they are recursively corrected in each calculation. Therefore, the traditional UKF will be very difficult to estimate the accurate values of the suspicious parameters. To solve this issue, the IAUKF [36] is introduced in this paper with some modifications thus applicable to solve the line parameters estimation problem based on the augmented state estimation. A brief introduction of the modified IAUKF is given as follows:

Let a nonlinear system be shown in (1).

$$
\left\{ \begin{array}{l} x _ {k} = f \left(x _ {k - 1}\right) + w _ {k} \\ z _ {k} = h \left(x _ {k}\right) + v _ {k} \end{array} \right. \tag {1}
$$

where  $x_{k}$  and  $z_{k}$  are the  $n$  dimensional state vector and the  $m$  dimensional measurement vector at time step  $k$ , respectively;  $w_{k}$  and  $v_{k}$  are the process noise vector and measurement noise vector respectively where  $w_{k} \sim N(0, \mathbf{Q}_{k})$  and  $v_{k} \sim N(0, \mathbf{R}_{k})$ ;  $f(\cdot)$  is the state equation and  $h(\cdot)$  is the measurement equation.

The UKF selects a number of sigma points through the unscented transform (UT) to apply to the Kalman filter (KF) framework. It can be divided into three stages: prediction, correction and the process noise variance matrix estimation. The execution procedures are described in the following.

Stage 1. Prediction:

$$
\left\{ \begin{array}{l} X _ {k - 1} ^ {(0)} = \hat {x} _ {k - 1} \\ X _ {k - 1} ^ {(i)} = \hat {x} _ {k - 1} + \left(\sqrt {(n + \lambda) P _ {k - 1}}\right) _ {i}, i = 1, 2, \dots , n \\ X _ {k - 1} ^ {(i)} = \hat {x} _ {k - 1} - \left(\sqrt {(n + \lambda) P _ {k - 1}}\right) _ {i}, i = n + 1, \dots 2 n \end{array} \right. \tag {2}
$$

$$
\left\{ \begin{array}{l} W _ {0} ^ {(m)} = \frac {\lambda}{n + \lambda} \\ W _ {0} ^ {(c)} = \frac {\lambda}{n + \lambda} + (1 - a ^ {2} + \beta) \\ W _ {i} ^ {(m)} = W _ {i} ^ {(c)} = \frac {1}{2 (n + \lambda)}, i = 1, 2, \dots , 2 n \end{array} \right. \tag {3}
$$

where  $\hat{x}_{k - 1}$  and  $P_{k - 1}$  are the state estimation result and the estimated error covariance matrix at time step  $k - 1$ , respectively; The fine-tuning parameter  $\lambda = a^{2}(n + \kappa) - n$  is used to control

the point-to-mean distance;  $n$  is the state vector dimension;  $a$  is the proportional correction factor and the commonly used values is  $10^{-4} \leq a \leq 1$  for Gaussian distribution;  $\kappa$  is the secondary sampling factor and its value is usually taken 0 or  $3 - n$  [40];  $\beta$  is the candidate parameter and  $\beta = 2$  is optimal for the Gaussian distribution [41];  $(\sqrt{(n + \lambda)P_{k-1}})_i$  denotes the  $i$ th column of  $\sqrt{(n + \lambda)P_{k-1}}$ ;  $W_i^{(m)}$  and  $W_i^{(c)}$  are the  $i$ th mean and variance calculation weights, respectively.

$$
x _ {k \mid k - 1} ^ {(i)} = f \left(X _ {k - 1} ^ {(i)}\right), i = 0, 1, \dots , 2 n \tag {4}
$$

$$
\hat {x} _ {k \mid k - 1} = \sum_ {i = 0} ^ {2 n} W _ {i} ^ {(m)} x _ {k \mid k - 1} ^ {(i)} \tag {5}
$$

$$
\begin{array}{l} P _ {k | k - 1} = \sum_ {i = 0} ^ {2 n} W _ {i} ^ {(c)} \left[ x _ {k | k - 1} ^ {(i)} - \hat {x} _ {k | k - 1} \right] \\ \times \left[ x _ {k \mid k - 1} ^ {(i)} - \hat {x} _ {k \mid k - 1} \right] ^ {\mathrm {T}} + \mathbf {Q} _ {k} \tag {6} \\ \end{array}
$$

where  $x_{k|k-1}^{(i)}$  is the sigma point propagated through the state equation;  $\hat{x}_{k|k-1}$  is the state prediction value obtained by propagation;  $P_{k|k-1}$  is the prediction covariance matrix of the state variable.

Stage 2. Correction:

$$
\left\{ \begin{array}{l} X _ {k | k - 1} ^ {(0)} = \hat {x} _ {k | k - 1} \\ X _ {k | k - 1} ^ {(i)} = \hat {x} _ {k | k - 1} + \left(\sqrt {(n + \lambda) P _ {k | k - 1}}\right) _ {i}, i = 1, 2, \dots , n \\ X _ {k | k - 1} ^ {(i)} = \hat {x} _ {k | k - 1} - \left(\sqrt {(n + \lambda) P _ {k | k - 1}}\right) _ {i}, i = n + 1, \dots , 2 n \end{array} \right. \tag {7}
$$

$$
Z _ {k \mid k - 1} ^ {(i)} = h \left(X _ {k \mid k - 1} ^ {(i)}\right), i = 1 = 0, 1, \dots , 2 n \tag {8}
$$

$$
\hat {z} _ {k \mid k - 1} = \sum_ {i = 0} ^ {2 n} W _ {i} ^ {(m)} Z _ {k \mid k - 1} ^ {(i)} \tag {9}
$$

$$
\begin{array}{l} P _ {z z, k | k - 1} = \sum_ {i = 0} ^ {2 n} W _ {i} ^ {(c)} \left[ Z _ {k | k - 1} ^ {(i)} - \hat {z} _ {k | k - 1} \right] \\ \times \left[ Z _ {k \mid k - 1} ^ {(i)} - \hat {z} _ {k \mid k - 1} \right] ^ {\mathrm {T}} + \mathbf {R} _ {k} \tag {10} \\ \end{array}
$$

$$
\begin{array}{l} P _ {x z, k \mid k - 1} = \sum_ {i = 0} ^ {2 n} W _ {i} ^ {(c)} \left[ X _ {k \mid k - 1} ^ {(i)} - \hat {x} _ {k \mid k - 1} \right] \\ \times \left[ Z _ {k | k - 1} ^ {(i)} - \hat {z} _ {k | k - 1} \right] ^ {\mathrm {T}} \tag {11} \\ \end{array}
$$

where  $Z_{k|k-1}^{(i)}$  is the sigma point propagated through the measurement equation at time step  $k$ ;  $\hat{z}_{k|k-1}$  is the measured prediction value obtained by the propagation;  $P_{zz,k|k-1}$  and  $P_{xz,k|k-1}$  are the covariance matrix and the cross-covariance matrix of the predicted measurement, respectively.

$$
K _ {k} = P _ {x z, k \mid k - 1} P _ {z z, k \mid k - 1} ^ {- 1}. \tag {12}
$$

$$
\hat {x} _ {k} = \hat {x} _ {k \mid k - 1} + K _ {k} \left(z _ {k} - \hat {z} _ {k \mid k - 1}\right). \tag {13}
$$

$$
P _ {k} = P _ {k \mid k - 1} - K _ {k} P _ {z z, k \mid k - 1} K _ {k} ^ {\mathrm {T}}. \tag {14}
$$

The estimated state vector  $\hat{x}_k$  is obtained and the covariance matrix  $P_{k}$  is updated after the stage of correction.

Stage 3. The process noise variance matrix estimation:

To perceive  $\mathbf{Q}_k$  in the process of the UKF, it is necessary to estimate its value by embedding the NSE [36] into the UKF.

$$
\varepsilon_ {k} = z _ {k} - \hat {z} _ {k | k - 1} \tag {15}
$$

$$
d _ {k} = (1 - b) / \left(1 - b ^ {k + 1}\right) \tag {16}
$$

$$
\begin{array}{l} \mathbf {Q} _ {k + 1} = (1 - d _ {k}) Q _ {k} + d _ {k} \left[ K _ {k} \varepsilon_ {k} \varepsilon_ {k} ^ {\mathrm {T}} K _ {k} ^ {\mathrm {T}} + P _ {k} \right. \\ \left. - \sum_ {i = 0} ^ {2 n} W _ {i} ^ {(c)} \left(X _ {k \mid k - 1} ^ {(i)} - \hat {x} _ {k \mid k - 1}\right) \left(X _ {k \mid k - 1} ^ {(i)} - \hat {x} _ {k \mid k - 1}\right) ^ {\mathrm {T}} \right] \tag {17} \\ \end{array}
$$

$$
\mathbf {Q} _ {k + 1} = \left\{ \begin{array}{l} \mathbf {Q} _ {k + 1}, \text {i f} \mathbf {Q} _ {k + 1} \text {i s n o n n e g a t i v e d e f i n i t e} \\ \left(1 - d _ {k}\right) \mathbf {Q} _ {k} + d _ {k} \left[ \operatorname {d i a g} \left(K _ {k} \varepsilon_ {k} \varepsilon_ {k} ^ {\mathrm {T}} K _ {k} ^ {\mathrm {T}}\right) \right. \\ \left. + K _ {k} P _ {z z, k | k - 1} K _ {k} ^ {\mathrm {T}} \right], \text {o t h e r w i s e} \end{array} \right. \tag {18}
$$

where  $\varepsilon_{k}$  is the residual between the system measurement and the predicted measurement at time step  $k$ ;  $b$  is the forgetting factor and  $0.95 \leq b \leq 0.995$ , which is set to be 0.96 in this paper;  $\text{diag}()$  is the function returning a diagonal matrix of the same dimension as the operation matrix that is made up of the diagonal elements of the operated matrix.  $\mathbf{Q}_{k+1}$  is recalculated by the biased NSE in (18) if its value calculated by the unbiased NSE in (17) loses the nonnegative definiteness.

# IV. PROPOSED AUGMENTED STATE-SPACE MODEL

To estimate the suspicious parameters by the augmented state estimation, the distribution network augmented state-space model needs to be established. The state equation of suspicious line parameters can be established based on their time-invariant characteristics. The measurement equation is modified with the consideration of the suspicious parameters. The developed augmented state-space model for dynamic state estimation based on the IAUKF is developed by combining the state equation of the system state variables (voltage amplitude and phase angle of each bus), the parameter state equation and the modified measurement equation. Since the line parameters are constant over a certain period and can be estimated off-line, the augmented state-space model can be extended to the form under multiple measurement snapshots to avoid the failure of estimation due to the insufficient measurement redundancy.

# A. Distribution Network State-Space Model

The distribution network state variable  $x_{k} = [V_{i},\delta_{i}]^{\mathrm{T}}$  is the voltage amplitude and phase angle of each bus at time step  $k$ . Its state equation is established by Holt's dual exponential smoothing method. The distribution network state equation  $f()$  can be expressed as (19).

$$
\left\{ \begin{array}{l} x _ {k \mid k - 1} = S _ {k - 1} + b _ {k - 1} \\ S _ {k - 1} = \alpha_ {\mathrm {H}} x _ {k - 1} + (1 - \alpha_ {\mathrm {H}}) x _ {k - 1 \mid k - 2} \\ b _ {k - 1} = \beta_ {\mathrm {H}} \left(S _ {k - 1} - S _ {k - 2}\right) + (1 - \beta_ {\mathrm {H}}) b _ {k - 2} \end{array} \right. \tag {19}
$$

where  $x_{k-1}$  represents the state value of the distribution network at time step  $k-1$ ;  $x_{k|k-1}$  indicates the state prediction value at time step  $k$  obtained by the state equation;  $S_{k-1}$  and  $b_{k-1}$  are the horizontal and vertical components, respectively;  $\alpha_{\mathrm{H}}$  and  $\beta_{\mathrm{H}}$  are called the smoothing parameter and their value are usually taken in [0, 1]. In this work, the smoothing parameters of  $\alpha_{\mathrm{H}} = 0.8$  and  $\beta_{\mathrm{H}} = 0.5$  are adopted, as suggested in [42].

Based on the measurement type of SCADA and PMU, the measurement vector is composed as (20) in the distribution network. The voltage phasor measurement can only be obtained at the buses configured with the PMU.

$$
z _ {k} = \left[ V _ {i}, P _ {i}, Q _ {i}, P _ {i j}, Q _ {i j}, \delta_ {i} ^ {(P M U)} \right] ^ {\mathrm {T}} \tag {20}
$$

where  $V_{i}$  and  $\delta_{i}^{(PMU)}$  are the voltage amplitude and phase angle of the bus  $i$ , respectively;  $P_{i}$  and  $Q_{i}$  are the injected active and reactive power of the bus, respectively;  $P_{ij}$  and  $Q_{ij}$  are the active and reactive power of the branch  $i - j$ .

Associating with the measurement types in  $z_{k}$ , the measurement equation  $h()$  is as follows:

$$
\left\{ \begin{array}{l} V _ {i} = V \\ P _ {i} = V _ {i} \sum_ {j} V _ {j} \left(G _ {i j} \cos \delta_ {i j} + B _ {i j} \sin \delta_ {i j}\right) \\ Q _ {i} = V _ {i} \sum_ {j} V _ {j} \left(G _ {i j} \sin \delta_ {i j} - B _ {i j} \cos \delta_ {i j}\right) \\ P _ {i j} = V _ {i} ^ {2} g _ {i j} - V _ {i} V _ {j} g _ {i j} \cos \delta_ {i j} - V _ {i} V _ {j} b _ {i j} \sin \delta_ {i j} \\ Q _ {i j} = - V _ {i} ^ {2} b _ {i j} - V _ {i} V _ {j} g _ {i j} \sin \delta_ {i j} + V _ {i} V _ {j} b _ {i j} \cos \delta_ {i j} \\ \delta_ {i} ^ {(P M U)} = \delta_ {i} \end{array} \right. \tag {21}
$$

where  $G_{ij}$  and  $B_{ij}$  are the real and imaginary parts of the element of the  $i$ th row and  $j$ th column of the node admittance matrix, respectively;  $g_{ij}$  and  $b_{ij}$  are the conductance and susceptance of the branch  $i - j$ , respectively;  $\delta_{ij} = \delta_i - \delta_j$ .

Combining the state equation with the measurement equation, the state-space model can be established as (22).

$$
\left\{ \begin{array}{l} x _ {k} = f \left(x _ {k - 1}\right) + w _ {x \mid k} \\ z _ {k} = h \left(x _ {k}\right) + v _ {k} \end{array} \right. \tag {22}
$$

where  $w_{x|k}$  is the process noise vector of the state variable at time step  $k$  and  $w_{x|k} \sim N(0, \mathbf{Q}_{x|k})$ ;  $v_k$  is the measurement noise vector and  $v_k \sim N(0, \mathbf{R}_k)$ ;  $\mathbf{R}_k$  can be obtained by the nameplate data of the measurement equipment as mentioned before;  $\mathbf{Q}_{x|k}$  changes with the fluctuation of the load, therefore it is unknown and time-varying. Hence, it is necessary to be estimated by the NSE of IAUKF.

# B. Distribution Network Augmented State-Space Model

The suspicious parameters are set as unknown parameter variables. The augmented state vector is constituted by combining the unknown parameter state variables and the system state variables as  $(23)\sim (25)$ .

$$
\bar {x} _ {k} = \left[ x _ {k} p _ {k} \right] ^ {\mathrm {T}} \tag {23}
$$

$$
x _ {k} = \left[ V _ {i}, \delta_ {i} \right] ^ {\mathrm {T}} \tag {24}
$$

$$
p _ {k} = \left[ r _ {1}, x _ {1}, r _ {2}, x _ {2}, \dots , r _ {p}, x _ {p} \right] ^ {\mathrm {T}} \tag {25}
$$

where the augmented state vector  $\bar{x}_k$  includes two parts: the system state vector  $x_k$  and the parameter state vector  $p_k$ ;  $p_k$  consists of  $2p$  dimensional resistance and reactance parameters when the number of suspicious lines is  $p$ .

Similar to the state transition equation of synchronous machine [32], [34], the state equation of line parameters can be given as (26). The intuition behind (26) is that the line parameters of the time step  $k$  and  $k - 1$  should be equal when the parameter values are accurately estimated.

$$
p _ {k} = p _ {k - 1} + w _ {p \mid k} \tag {26}
$$

where  $w_{p|k}$  is the process noise vector of the parameter variables at time step  $k$ , which is assumed to be Gaussian white noise to be applicable to the framework of IAUKF, i.e.,  $w_{p|k} \sim N(0, \mathbf{Q}_{p|k})$ . The parameters are uncorrelated with each other.

The augmented state equation can be established by combining the system state equation (19) with the parameter state equation (26), as shown in (27).

$$
\bar {x} _ {k} = \left[ \begin{array}{c} f (x _ {k - 1}) \\ p _ {k - 1} \end{array} \right] + \left[ \begin{array}{c} w _ {x | k} \\ w _ {p | k} \end{array} \right]. \tag {27}
$$

The process noise variance matrix of the augmented state equation is a block diagonal matrix as shown in (28) since the state vector of the system and the parameter vector are uncorrelated.

$$
\mathbf {Q} _ {k} = \left[ \begin{array}{c c} \mathbf {Q} _ {x \mid k} & 0 \\ 0 & \mathbf {Q} _ {p \mid k} \end{array} \right]. \tag {28}
$$

The measurement equation reflects the relationship between the measurement and the state vector, and its construction requires line parameters. When the suspicious parameters are considered as unknown parameter variables like system state variables, the measurement equation need to be modified to the form considering parameter variables as (29).

$$
z _ {k} = h \left(x _ {k}, p _ {k}\right) + v _ {k}. \tag {29}
$$

Through combining (27) and (29), the augmented state-space model can be obtained as follows:

$$
\left\{ \begin{array}{l} \bar {x} _ {k} = \left[ \begin{array}{c} f (x _ {k - 1}) \\ p _ {k - 1} \end{array} \right] + \left[ \begin{array}{c} w _ {x | k} \\ w _ {p | k} \end{array} \right] \\ z _ {k} = h (x _ {k}, p _ {k}) + v _ {k} \end{array} \right. \tag {30}
$$

The process noise of the parameter variables reflects the deviation between the true parameter values and the current parameter estimation values. Since the true parameter values are unknown, and the measurement will correct the parameter estimation values at each time step, this deviation is unknown and time-varying. Hence, the process noise variance matrix  $\mathbf{Q}_{p|k}$  of the parameter variables cannot be assumed constant and requires estimation during the parameter estimation.

# C. Distribution Network Augmented State-Space Model Under Multiple Measurement Snapshots
![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-28/7f406345-5a93-423a-8be2-3cd72cf2cd67/ef1e4d71e692716ea3750e1c193e490f95aa784bd9c5865bb261d84703e81111.jpg)
Fig. 2. The structure of the augmented state vector.

The line parameters estimation can be implemented off-line, and the line parameters remain constant over a certain period. In addition, as shown in (31), the measurement redundancy can be effectively improved by applying multiple measurement snapshots data. Therefore, the augmented state-space model can be extended to the form under multiple measurement snapshots to avoid the failure of estimation caused by insufficient measurement redundancy.
$$
\frac {m t}{n t + n _ {p}} = \frac {m}{n + n _ {p} / t} > \frac {m}{n + n _ {p}} \tag {31}
$$
where  $n_p, t, m$  and  $n$  are the dimension of the unknown parameters, the number of the used multiple measurement snapshots, the measurement and system state vector dimension, respectively.

The state and measurement vector need to be reconstructed when using data under multiple measurement snapshots. Herein, the augmented state vector  $\bar{X}_k$  is the combination of the system state vector of multiple time steps and the unknown parameter state vector, as shown in (32). Also, the measurement vector  $Z_k$  also needs to be reconstructed correspondingly as (34).

$$
\bar {X} _ {k} = \left[ X _ {k} p _ {k} \right] ^ {\mathrm {T}} \tag {32}
$$

$$
X _ {k} = \left[ x _ {1}, x _ {2}, \dots , x _ {t} \right] \tag {33}
$$

$$
Z _ {k} = \left[ z _ {1}, z _ {2}, \dots , z _ {t} \right] ^ {\mathrm {T}} \tag {34}
$$

where  $x_{t}$  and  $z_{t}$  are the system state and measurement vector at time step  $t$ , respectively.

The augmented state vector at each step is shown in Fig. 2. It can be seen from Fig. 2 that the part of the system state vector under multiple measurement snapshots between two steps are still adjacent time steps at the corresponding positions. Hence, the state equation under multiple measurement snapshots can be obtained by expanding (27) as shown in (35).

$$
\bar {X} _ {k} = \left[ \begin{array}{c} f (X _ {k - 1}) \\ p _ {k - 1} \end{array} \right] + \left[ \begin{array}{c} w _ {X | k} \\ w _ {p | k} \end{array} \right]. \tag {35}
$$

Also, the measurement equation under multiple measurement snapshots can be obtained by expanding (29) as follows:

$$
Z _ {k} = \left[ \begin{array}{c} z _ {1} \\ z _ {2} \\ \vdots \\ z _ {t} \end{array} \right] = \left[ \begin{array}{c} h (x _ {1}, p _ {k}) \\ h (x _ {2}, p _ {k}) \\ \vdots \\ h (x _ {t}, p _ {k}) \end{array} \right] + \left[ \begin{array}{c} v _ {1 | k} \\ v _ {2 | k} \\ \vdots \\ v _ {t | k} \end{array} \right]. \tag {36}
$$

Correspondingly, the process and measurement noise variance matrix are shown in (37) and (38), respectively.

$$
\mathbf {Q} _ {k} = \operatorname {d i a g} \left(\mathbf {Q} _ {x \mid 1}, \mathbf {Q} _ {x \mid 2}, \dots , \mathbf {Q} _ {x \mid t}, \mathbf {Q} _ {p \mid k}\right). \tag {37}
$$


TABLEI THE CONFIGURATION PARAMETERS OF MEASUREMENT


<table><tr><td>Measurement type</td><td>Bus configuration</td></tr><tr><td>SCADA</td><td>1,2,4,5,7,8,10,12,13,15,16,18,20,21,23,25,27,28,30,31,33</td></tr><tr><td>PMU</td><td>3,6,9,11,14,17,19,22,24,26,29,32</td></tr></table>

$$
\mathbf {R} _ {k} = \operatorname {d i a g} \left(\mathbf {R} _ {1}, \mathbf {R} _ {2}, \dots , \mathbf {R} _ {t}\right). \tag {38}
$$

# V. SIMULATION EXPERIMENTS AND NUMERICAL RESULTS

In this work, the IEEE 33-bus system [43] and the 118-bus system [44] constructed in MATPOWER [45] are adopted to verify the effectiveness of the proposed method. To have more realistic case studies, the measurement data used in the simulation are obtained by adding Gaussian noise into the results of power flow [12]. And the power flow is calculated by MATPOWER. The standard deviation of SCADA measurement is set to 0.02. The standard deviations of PMU voltage amplitude and phase angle are set to be 0.005 and 0.002, respectively. The parameters required for the UKF are:  $a = 0.001$ ,  $\kappa = 0$  and  $\beta = 2$ . And the initial  $\mathbf{Q}_k$  is set to be  $10^{-6}\mathbf{I}$  [32], [35] which will be adaptive during the parameter estimation. Because the true values of suspicious parameters are unknown, the initial values are set to be a small value (e.g., 0.01 or 0.02). The difference between the two adjacent steps estimation results will become small when the results tend to converge. To determine the convergence step, the convergence criterion is established as (39).

$$
\left| p _ {k} - p _ {k - 1} \right| \leq \delta \tag {39}
$$

where  $\delta$  is the accuracy requirement which is set to be 0.001 in this paper.

When both the resistance and reactance parameters satisfy the convergence criteria, the average values of all parameter estimation results after convergence are obtained as the final estimation results, as shown in (40).

$$
\hat {p} = \frac {\sum_ {k = n} ^ {N} p _ {k}}{L} \tag {40}
$$

where  $n$  is the step of convergence;  $N$  is the total number of steps and  $L = N - n + 1$ .

# A. Simulations on the IEEE 33-Bus System

The effectiveness of the proposed method is first verified in the case that there are suspicious line parameters of a single branch of the IEEE 33-bus system. The measurement configuration is shown in Table I. It is assumed that the parameters of branch 3-4 are unknown. Thus, its resistance and reactance are treated as state variables and are augmented into the state vector. After the augmented vector is estimated by the IAUKF for 200-time steps under single measurement snapshot, the result curves of the parameters are shown in Fig. 3. It can be seen from Fig. 3 that both the resistance and reactance of the branch converge to straight lines after estimation in a certain number of time steps. To improve the estimation accuracy, the final estimation results are obtained by (39) and (40). The relative errors of the final results of resistance and reactance are  $0.18\%$  and  $1.55\%$ , respectively. Both the resistance and reactance can be accurately estimated.

To further validate the effectiveness of the proposed method for the parameters of each branch under single measurement snapshot, the parameters of other 31 branches are set as unknown and estimated by the IAUKF respectively. After a certain number of time steps of performing, all parameters, except the parameters of the four end branches (17-18, 21-22, 24-25 and 32-33) can converge to straight lines and obtain well-estimated results.

The estimation results of the four end branches can't converge under single measurement snapshot, such as the estimation results of the parameters of the branch 21-22 shown in Fig. 4. Even though 350 time-steps have been recursively estimated, they still do not converge. This is because of the insufficient measurement redundancy for end branches when estimation. Therefore, to increase the measurement redundancy and enable the estimation of the parameters of end branches, the augmented state-space model under multiple measurement snapshots is adopted. The number of the measurement snapshots is set to be 5, and the estimation results of branch 21-22 are shown in Fig. 5. The parameters of the branch converge to straight lines after estimation in a certain number of steps by increasing the measurement redundancy. The relative errors of the final results under multiple measurement snapshots are  $0.36\%$  and  $2.96\%$ . The results show that the augmented state-space model under multiple measurement snapshots can effectively improve the measurement redundancy and avoid the failure of estimation caused by insufficient measurement redundancy. Thus, when the parameters of multi-branch are suspicious, the accurate estimation results can be obtained by using the model under multiple measurement snapshots.

It is assumed that the parameters of 4 branches including the end branches are suspicious to analyze the estimation performance when the parameters of multiple branches are unknown. The 8 unknown parameters are augmented into the state vector for the model under multiple measurement snapshots. And, the number of the measurement snapshots still is 5. The estimation results are shown in Fig. 6. As can be seen from Fig. 6, the 8 parameters can converge to estimated values. The final estimation values are shown in Table II. According to the table, the relative errors of resistance and reactance of estimated values for branch 3-4 are  $0.13\%$  and  $0.09\%$  which are more accurate than those under single snapshot (noting that the relative errors of resistance and reactance are  $0.18\%$  and  $1.55\%$ , respectively.). This indicates that the improvement of measurement redundancy not only guarantees the accurate estimation of the end branch parameters but also can obtain the more accurate estimated values of other branches. In summary, the proposed method in this paper can obtain accurate estimated values when both single-branch and multi-branch parameters are unknown.



TABLE II ESTIMATION RESULTS OF FOUR BRANCHES
<table><tr><td colspan="2">Branch (bus # - bus #)</td><td>3-4</td><td>7-8</td><td>21-22</td><td>29-30</td></tr><tr><td rowspan="3">Resistance</td><td>True value (p.u)</td><td>0.2284</td><td>0.4439</td><td>0.4423</td><td>0.3166</td></tr><tr><td>Estimated value (p.u)</td><td>0.2281</td><td>0.4436</td><td>0.4446</td><td>0.3157</td></tr><tr><td>Estimation error (%)</td><td>0.13%</td><td>0.07%</td><td>0.52%</td><td>0.28%</td></tr><tr><td rowspan="3">Reactance</td><td>True value (p.u)</td><td>0.1163</td><td>0.1467</td><td>0.5848</td><td>0.1613</td></tr><tr><td>Estimated value (p.u)</td><td>0.1162</td><td>0.1463</td><td>0.5729</td><td>0.1601</td></tr><tr><td>Estimation error (%)</td><td>0.09%</td><td>0.27%</td><td>2.03%</td><td>0.74%</td></tr></table>




The PMU configured in the distribution network can provide high precision measurements, which helps to improve the accuracy of parameter estimation. Fig. 7 and Fig. 8 show the comparison of the parameter estimation results with and without configuration of  $\mu$  PMU. According to the Fig. 7, the accuracy of the estimation results  $R_{3-4}^{PMU-single}$  and  $X_{3-4}^{PMU-single}$  under single measurement snapshot with the PMU is resembled as the results  $R_{3-4}^{SCADA-multi}$  and  $X_{3-4}^{SCADA-multi}$  under multiple measurement snapshots without the PMU while the results  $R_{3-4}^{SCADA-single}$  and  $X_{3-4}^{SCADA-single}$  have large errors. It indicates not only that the PMU can improve the estimation accuracy, but also that the improvement of the redundancy of the measurement can improve the estimation accuracy. Moreover, Fig. 8 indicates that the error of the results  $R_{21-22}^{SCADA}$  and  $X_{21-22}^{SCADA}$  without the PMU under multiple measurement snapshots is obviously higher than the results  $R_{21-22}^{PMU}$  and  $X_{21-22}^{PMU}$  with the PMU under multiple measurement snapshots. This demonstrates the effectiveness of PMU in improving estimation accuracy and the necessity of configuring  $\mu$  PMU in a large distribution network.

In order to test the necessity of perceiving  $\mathbf{Q}_k$  in the process of UKF, the UKF that keeps  $\mathbf{Q}_k$  as a constant is performed to estimate the line parameters of branch 3-4 under single and multiple measurement snapshots. As evident from Fig. 9 and Fig. 10, the estimated results fail to converge to the true values in the two cases. In contrast, the estimation results of the method proposed in this paper can quickly converge to the true values under single measurement snapshot as shown in Fig. 3.

# B. Simulations on the 118-Bus System

To further validate the performance of the proposed method in this paper, the 118-bus system with DGs is adopted. The total load of the system is  $22709.7\mathrm{kW} + \mathrm{j}17041.1\mathrm{kVar}$ . The penetration rate of DGs is set to be  $20\%$ . And, the 118-bus system topology, measurement configuration and DG configuration are shown Fig. 11. The rated capacities of PV1, PV2 and WT are  $400\mathrm{kW} + \mathrm{j}132\mathrm{kVar}$ ,  $600\mathrm{kW} + \mathrm{j}196\mathrm{kVar}$  and  $300\mathrm{kW} + \mathrm{j}90\mathrm{kVar}$ , respectively. The conclusion obtained from the simulation results of the 118-bus system with DGs is consistent with those in the IEEE 33-bus system. For instance, it can be seen from Fig. 12 that the result curves of branch 64-78 under single measurement snapshot can converge to estimated values. However, the result curves of end branch 16-17 under single measurement snapshot can't converge as shown in Fig. 13. Consistent with the IEEE 33-bus system, the model under multiple measurement snapshots (noting that the number of the snapshots is set to 10) can give accurate estimated values of the parameters of branch 16-17 according to Fig. 14.

The parameters of 21 randomly selected branches are considered to be unknown. These 42 parameters are augmented to the state vector under multiple measurement snapshots and estimated simultaneously. The relative errors of the estimated results for all branch resistances and reactances are shown in Fig. 15. All parameters can be estimated well. The mean relative errors of  $R$  and  $X$  are  $0.18\%$  and  $0.27\%$ , respectively. These results show that the method proposed in this paper performs well in the case of large systems with DGs and with unknown multi-branch parameters.

The model under multiple measurement snapshots effectively improves the measurement redundancy, which will enhance the robustness of the proposed method for bad data. Bad data mainly include erroneous data with large errors and missing data due to communication failures. The estimation performance of the parameters of 21 branches is evaluated by considering the case of  $10\%$  of measurements are bad data at each time step, of which  $5\%$  are missing data and  $5\%$  are erroneous data. The erroneous data is obtained by adding 3 times of Gaussian noise into the results of power flow. The estimated results are shown in Fig. 16. The estimation results of the parameters of all branches yield good performance, i.e. the mean relative errors of  $R$  and  $X$  are  $1.72\%$  and  $1.65\%$ , respectively. The branch 16-17 is with higher relative errors that are still considered acceptable. The estimated results demonstrate the robustness of the proposed model under multiple measurement snapshots.
