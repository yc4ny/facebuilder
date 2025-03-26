<p align="center">

  <h1 align="center">Facebuilder Tool</h1>
  <p align="center">
    <a href="https://www.cubox.ai/"><strong>CUBOX AI Lab</strong></a><sup></sup>
  <div align="center">
    <img src="./assets/teaser.png" alt="Logo" width="100%">
  </div>

  <p align="center">
    <a href='https://github.com/yc4ny/facebuilder'>
      <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
    <a href="" target='_blank'>
      <img src="https://visitor-badge.laobi.icu/badge?page_id=yc4ny.facebuilder&left_color=gray&right_color=orange">
    </a>
  </p>
</p>

<br/>

**Facebuilder**
FaceBuilder is a sophisticated 3D face modeling tool developed by Keentools' that allows precise manipulation of 3D facial meshes using a pin-based deformation system. The tool supports interactive mesh editing across multiple views with synchronized 3D consistency.

Key Features

Multi-mode Pin System
FaceBuilder implements an intuitive pin-based interaction system with three specialized manipulation modes:


Single Pin (1-Pin) Mode
![1pin](assets/1pin.gif)

Rotation Mechanism: Mouse movements (dx, dy) are converted to rotation elements

Horizontal movement (dx) → Y-axis rotation
Vertical movement (dy) → X-axis rotation
These rotation values are combined with the existing rotation matrix for natural rotation
The entire mesh follows the mouse position precisely via calculated translation values
Ray casting techniques ensure 2D mouse input accurately reflects on the 3D mesh
Face structure maintains integrity while rotating naturally according to mouse movement


Dual Pin (2-Pin) Mode
![2pin](assets/2pin.gif)

Scaling Behavior:
Pulling pins outward → Enlarges the mesh
Pulling pins inward → Reduces the mesh


Non-dragged pin remains fixed in both mesh and image space
All rotation and scaling operations use the fixed pin as a reference point
Allows precise control of mesh size while maintaining proportions


Triple Pin (3-Pin) Mode
![3pin](assets/3pin.gif)
3D Rotation Control: Dragging the third pin while using two fixed pins as anchor points
Uses PnP (Perspective-n-Point) algorithm to calculate accurate 3D rotation and translation
Maintains proportions while enabling precise mesh control
Two non-dragged pins remain exactly fixed in their original positions

