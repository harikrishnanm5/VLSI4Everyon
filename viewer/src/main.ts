/**
 * OpenProcess3D Viewer Main Application.
 *
 * Initializes Three.js scene, camera, controls, lighting, and renders
 * semiconductor process layers from scene.json with Z-axis up orientation.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadScene } from './loader';

// Set standard 3D semiconductor CAD convention: Z axis is vertical up
THREE.Object3D.DEFAULT_UP = new THREE.Vector3(0, 0, 1);

async function init(): Promise<void> {
  const container = document.getElementById('canvas-container');
  const infoOverlay = document.getElementById('info-overlay');
  const errorOverlay = document.getElementById('error-overlay');

  if (!container) {
    throw new Error("Missing '#canvas-container' element in DOM");
  }

  // 1. Scene setup
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  // 2. Camera setup (Z-axis up)
  const width = container.clientWidth || window.innerWidth;
  const height = container.clientHeight || window.innerHeight;
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(20, -25, 20);

  // 3. Renderer setup
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  container.appendChild(renderer.domElement);

  // 4. OrbitControls setup
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = true;

  // 5. Lighting setup
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambientLight);

  const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
  mainLight.position.set(1.0, -1.5, 2.0).normalize();
  scene.add(mainLight);

  const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
  fillLight.position.set(-1.0, 1.5, -0.5).normalize();
  scene.add(fillLight);

  // Helper axes (RGB = XYZ)
  const axesHelper = new THREE.AxesHelper(5);
  scene.add(axesHelper);

  // 6. Framing function
  function frameCameraToBounds(bounds: THREE.Box3, offsetFactor = 1.8): void {
    const center = new THREE.Vector3();
    bounds.getCenter(center);

    const size = new THREE.Vector3();
    bounds.getSize(size);

    const maxDim = Math.max(size.x, size.y, size.z, 1.0);
    const fovRad = (camera.fov * Math.PI) / 180;
    let distance = (maxDim / 2) / Math.tan(fovRad / 2);
    distance *= offsetFactor;

    // View from an isometric angle looking toward center
    const viewDirection = new THREE.Vector3(1.0, -1.3, 0.9).normalize();
    camera.position.copy(center).addScaledVector(viewDirection, distance);

    camera.near = Math.max(0.01, distance / 100);
    camera.far = Math.max(200, distance * 50);
    camera.updateProjectionMatrix();

    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();

    // Resize axes helper to match model scale
    axesHelper.scale.setScalar(Math.max(1.0, maxDim * 0.2));
  }

  // 7. Load scene from ?scene=<url>, defaulting to /scene.json
  const urlParams = new URLSearchParams(window.location.search);
  const sceneUrl = urlParams.get('scene') || '/scene.json';

  try {
    if (infoOverlay) {
      infoOverlay.textContent = `Loading scene from ${sceneUrl}...`;
    }

    const loaded = await loadScene(sceneUrl);
    scene.add(loaded.group);
    frameCameraToBounds(loaded.bounds);

    if (infoOverlay) {
      const layerNames = loaded.layers.map((l) => l.name).join(', ');
      infoOverlay.innerHTML = `<strong>OpenProcess3D Viewer</strong><br/>Layers (${loaded.layers.length}): ${layerNames}`;
    }
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error('Failed to load scene:', err);
    if (errorOverlay) {
      errorOverlay.textContent = `Error loading scene: ${errorMsg}`;
      errorOverlay.style.display = 'block';
    }
    if (infoOverlay) {
      infoOverlay.textContent = 'Error loading scene';
    }
  }

  // 8. Resize handling
  window.addEventListener('resize', () => {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  // 9. Animation loop
  function animate(): void {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  requestAnimationFrame(animate);
}

// Start viewer application
init().catch((err) => {
  console.error('Fatal initialization error:', err);
});

