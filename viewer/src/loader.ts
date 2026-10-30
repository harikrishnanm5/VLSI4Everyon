/**
 * OpenProcess3D Scene Loader.
 *
 * Fetches and validates scene.json models and constructs Three.js meshes
 * from layer geometry definitions.
 */

import * as THREE from 'three';

export interface LayerData {
  name: string;
  material: string;
  z: number;
  thickness: number;
  approx: boolean;
  vertices: [number, number, number][];
  faces: [number, number, number][];
}

export interface BoundsData {
  min: [number, number, number];
  max: [number, number, number];
}

export interface SceneData {
  version: number;
  units: string;
  layers: LayerData[];
  bounds: BoundsData;
}

export interface LoadedLayer {
  name: string;
  material: string;
  z: number;
  thickness: number;
  approx: boolean;
  mesh: THREE.Mesh;
}

export interface SceneResult {
  data: SceneData;
  layers: LoadedLayer[];
  bounds: THREE.Box3;
  group: THREE.Group;
}

/** Distinct color palette for semiconductor layers. */
export const DEFAULT_LAYER_PALETTE: number[] = [
  0x4caf50, // nwell / silicon green
  0xff7043, // poly orange
  0x26c6da, // local interconnect cyan
  0x42a5f5, // met1 blue
  0xab47bc, // purple
  0xffca28, // amber
  0x26a69a, // teal
  0xec407a, // pink
  0x8d6e63, // brown
  0x78909c, // slate blue
];

/** Standard colors for well-known semiconductor layer names. */
export const KNOWN_LAYER_COLORS: Record<string, number> = {
  nwell: 0x4caf50,
  pwell: 0x66bb6a,
  diff: 0x81c784,
  poly: 0xff7043,
  li1: 0x26c6da,
  met1: 0x42a5f5,
  met2: 0x5c6bc0,
  met3: 0x7e57c2,
  met4: 0xab47bc,
  met5: 0xec407a,
};

/**
 * Validates the parsed JSON against the scene.json schema.
 */
export function validateScene(data: unknown): SceneData {
  if (typeof data !== 'object' || data === null) {
    raiseValidationError('Root must be an object');
  }

  const root = data as Record<string, unknown>;

  if (typeof root.version !== 'number') {
    raiseValidationError("Missing or invalid 'version' field");
  }

  if (root.version !== 1) {
    raiseValidationError(`Unsupported scene version: ${root.version}. Expected version 1`);
  }

  if (typeof root.units !== 'string' || root.units.trim() === '') {
    raiseValidationError("Missing or invalid 'units' field (expected string e.g. 'um')");
  }

  if (!Array.isArray(root.layers)) {
    raiseValidationError("'layers' field must be an array");
  }

  if (typeof root.bounds !== 'object' || root.bounds === null) {
    raiseValidationError("Missing or invalid 'bounds' object");
  }

  const boundsObj = root.bounds as Record<string, unknown>;
  validateVec3(boundsObj.min, 'bounds.min');
  validateVec3(boundsObj.max, 'bounds.max');

  const validatedLayers: LayerData[] = [];

  for (let idx = 0; idx < root.layers.length; idx++) {
    const rawLayer = root.layers[idx];
    const prefix = `layers[${idx}]`;

    if (typeof rawLayer !== 'object' || rawLayer === null) {
      raiseValidationError(`${prefix} must be an object`);
    }

    const l = rawLayer as Record<string, unknown>;

    if (typeof l.name !== 'string' || l.name.trim() === '') {
      raiseValidationError(`${prefix}.name must be a non-empty string`);
    }

    if (typeof l.material !== 'string') {
      raiseValidationError(`${prefix}.material must be a string`);
    }

    if (typeof l.z !== 'number' || Number.isNaN(l.z)) {
      raiseValidationError(`${prefix}.z must be a valid number`);
    }

    if (typeof l.thickness !== 'number' || Number.isNaN(l.thickness) || l.thickness <= 0) {
      raiseValidationError(`${prefix}.thickness must be a positive number`);
    }

    if (typeof l.approx !== 'boolean') {
      raiseValidationError(`${prefix}.approx must be a boolean`);
    }

    if (!Array.isArray(l.vertices)) {
      raiseValidationError(`${prefix}.vertices must be an array`);
    }

    const vertices: [number, number, number][] = [];
    for (let vIdx = 0; vIdx < l.vertices.length; vIdx++) {
      const v = l.vertices[vIdx];
      validateVec3(v, `${prefix}.vertices[${vIdx}]`);
      vertices.push(v as [number, number, number]);
    }

    if (!Array.isArray(l.faces)) {
      raiseValidationError(`${prefix}.faces must be an array`);
    }

    const faces: [number, number, number][] = [];
    for (let fIdx = 0; fIdx < l.faces.length; fIdx++) {
      const f = l.faces[fIdx];
      validateIndex3(f, `${prefix}.faces[${fIdx}]`, vertices.length);
      faces.push(f as [number, number, number]);
    }

    validatedLayers.push({
      name: l.name,
      material: l.material,
      z: l.z,
      thickness: l.thickness,
      approx: l.approx,
      vertices,
      faces,
    });
  }

  return {
    version: root.version as number,
    units: root.units as string,
    layers: validatedLayers,
    bounds: {
      min: boundsObj.min as [number, number, number],
      max: boundsObj.max as [number, number, number],
    },
  };
}

function raiseValidationError(message: string): never {
  throw new Error(`Scene validation error: ${message}`);
}

function validateVec3(v: unknown, path: string): void {
  if (!Array.isArray(v) || v.length !== 3 || !v.every((n) => typeof n === 'number' && !Number.isNaN(n))) {
    raiseValidationError(`${path} must be a 3-element array of numbers [x, y, z]`);
  }
}

function validateIndex3(f: unknown, path: string, vertexCount: number): void {
  if (
    !Array.isArray(f) ||
    f.length !== 3 ||
    !f.every((idx) => typeof idx === 'number' && Number.isInteger(idx) && idx >= 0 && idx < vertexCount)
  ) {
    raiseValidationError(
      `${path} must be a 3-element array of non-negative integer indices referencing existing vertices`
    );
  }
}

/**
 * Builds a THREE.Mesh from a LayerData definition using BufferGeometry.
 */
export function buildLayerMesh(layer: LayerData, color: THREE.ColorRepresentation): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();

  const vertexFlat = new Float32Array(layer.vertices.length * 3);
  for (let i = 0; i < layer.vertices.length; i++) {
    const v = layer.vertices[i];
    vertexFlat[i * 3 + 0] = v[0];
    vertexFlat[i * 3 + 1] = v[1];
    vertexFlat[i * 3 + 2] = v[2];
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(vertexFlat, 3));

  const faceFlat = new Uint32Array(layer.faces.length * 3);
  for (let i = 0; i < layer.faces.length; i++) {
    const f = layer.faces[i];
    faceFlat[i * 3 + 0] = f[0];
    faceFlat[i * 3 + 1] = f[1];
    faceFlat[i * 3 + 2] = f[2];
  }
  geometry.setIndex(new THREE.BufferAttribute(faceFlat, 1));

  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.35,
    metalness: 0.15,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = layer.name;
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  return mesh;
}

/**
 * Fetches and loads a scene.json from the specified URL.
 */
export async function loadScene(url: string): Promise<SceneResult> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch scene from "${url}": HTTP ${response.status} ${response.statusText}`);
  }

  const rawJson = await response.json();
  const sceneData = validateScene(rawJson);

  const group = new THREE.Group();
  group.name = 'scene_root';

  const loadedLayers: LoadedLayer[] = [];

  for (let idx = 0; idx < sceneData.layers.length; idx++) {
    const layerData = sceneData.layers[idx];
    const normalizedName = layerData.name.toLowerCase().trim();
    const color =
      KNOWN_LAYER_COLORS[normalizedName] ??
      DEFAULT_LAYER_PALETTE[idx % DEFAULT_LAYER_PALETTE.length];

    const mesh = buildLayerMesh(layerData, color);
    group.add(mesh);

    loadedLayers.push({
      name: layerData.name,
      material: layerData.material,
      z: layerData.z,
      thickness: layerData.thickness,
      approx: layerData.approx,
      mesh,
    });
  }

  const bounds = new THREE.Box3(
    new THREE.Vector3(...sceneData.bounds.min),
    new THREE.Vector3(...sceneData.bounds.max)
  );

  return {
    data: sceneData,
    layers: loadedLayers,
    bounds,
    group,
  };
}

