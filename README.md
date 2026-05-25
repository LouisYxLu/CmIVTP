# CmIVTP: Cross-modal Interaction-based Vessel Trajectory Prediction for Maritime Intelligence

Abstract
Maritime intelligent transportation systems (MITS) are essential for ensuring navigation safety and efficiency in busy waterways. However, accurate vessel trajectory prediction remains challenging due to the limitations of single-source data. Automatic identification system (AIS) data is often sparse or unavailable for small vessels, while closed-circuit television (CCTV) data alone cannot fully capture dynamic vessel behavior. To mitigate these challenges, we propose a cross-modal interaction-based vessel trajectory prediction (named CmIVTP) framework to model the intricate interactions between vessel dynamics and environmental constraints. Specifically, we introduce a target-aware scene encoder to extract scene semantic features, effectively capturing vessel-environment interactions and enhancing trajectory prediction accuracy. In addition, we propose a cross-modal interaction transformer, which integrates AIS-derived motion features, CCTV-based environmental features, and scene representations. It leverages cross-modal attention mechanisms to simultaneously capture intra-modal semantics and inter-modal interactions, ensuring dynamically consistent and environmentally feasible predictions. Furthermore, we construct a vessel group trajectory bank by clustering historical AIS trajectories into representative motion patterns, providing an efficient and scalable approach for candidate trajectory generation. Additionally, we introduce the maritime multimodal dataset plus (named Maritime-MmD$^+$), a large-scale dataset that synchronizes AIS data and CCTV video data, providing robust support for multimodal trajectory prediction research. Extensive experiments demonstrate that CmIVTP achieves better performance on multimodal-driven vessel trajectory prediction benchmarks.

Our data have been fully organized; however, due to certain restrictions, they cannot be made publicly available. If needed, they can be obtained by contacting the authors.

<div align="center">
  <img width="600" alt="Figure01" src="https://github.com/user-attachments/assets/3d1ccb99-1d96-4ac9-a700-728fe8eec87c" />
  <br>
  <p align="left" width="600">
    <br>
    <strong>Fig. 1</strong> The MITS integrates advanced infrastructure and artificial intelligence-driven analytics to enable cross-modal interaction-based vessel trajectory prediction (e.g., using AIS and CCTV data), ultimately enhancing maritime intelligence and MASS operations for safe, sustainable navigation.
  </p>
</div>

<div align="center">
  <img width="2000" alt="Figure02<img width="5327" height="1683" alt="Figure02" src="https://github.com/user-attachments/assets/7d0ad0d8-f233-4167-9cb0-9cc90e6c4ffa" />
" src="https://github.com/user-attachments/assets/3d1ccb99-1d96-4ac9-a700-728fe8eec87c" />
  <br>
  <p align="left" width="2000">
    <br>
    <strong>Fig. 1</strong> The flowchart of the proposed cross-modal interaction-based vessel trajectory prediction (named CmIVTP) framework. It consists of four main modules: the visual scene target-aware encoder (VSTaE) to extract environmental interaction features, the cross-modal interaction-based encoder (CmIE) to fuse AIS and CCTV data for modeling complex interactions, and the uncertainty-aware variational decoder (UaVD) to generate future trajectories. Additionally, a vessel group trajectory bank (VGTB) is constructed to improve the efficiency and accuracy of trajectory generation.
  </p>
</div>

