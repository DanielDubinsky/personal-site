---
hide:
  - navigation
  - toc
---

<div class="demo-page-header">
  <h1 class="demo-title">Open-Vocabulary Detection <span class="demo-title-x">x</span> Hailo-8</h1>
  <p class="demo-subtitle">Type any text query and get real-time object detections on edge hardware - no retraining, no recompilation.</p>
  <p class="demo-tagline">~24 FPS on Raspberry Pi 5 with Hailo-8.</p>
</div>

<video autoplay muted loop playsinline controls width="100%" style="border-radius:8px;">
  <source src="../../assets/videos/demo_open_vocab.mp4" type="video/mp4">
</video>

<div class="demo-details" markdown>

**How it works:** Edge AI accelerators freeze model weights at compile time. Current open-vocabulary methods bake text queries into the weights, requiring a full recompile for every new query. This system decouples the text encoder from the compiled model, allowing queries to be updated over the air in real time while inference keeps running.

In the video you can see queries being added and removed live - *person*, *bag*, *orange hat* - each with its own color-coded bounding boxes, updating continuously.

</div>
