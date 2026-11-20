/** The glTF JSON shapes and accessor reader moved to `@pmndrs/raytracer` (`packages/raytracer/src/gltf/read.ts`); bench modules keep importing them from here. */
export { gltfReadAccessor as readAccessor, gltfLocalMatrix as localMatrix, gltfRootNodes as rootNodes, GLTF_COMPONENT as COMPONENT, GLTF_NCOMP as NCOMP } from '@pmndrs/raytracer';
export type { GltfAccessor, GltfBufferView, GltfPrimitive, GltfNode, GltfImage, GltfSampler, GltfTexture, GltfCameraJson, GltfLightJson, GltfJson } from '@pmndrs/raytracer';
