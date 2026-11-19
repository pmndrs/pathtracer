"""Run from Blender in background mode. Prepare a compact glTF and camera sidecar for each hero scene."""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

assets = Path(__file__).resolve().parents[1] / "assets"
kind = sys.argv[sys.argv.index("--") + 1]
output = assets / "generated" / "hero-source"
output.mkdir(parents=True, exist_ok=True)


def web(v):
    return [v.x, v.z, -v.y]


def diffuse_image(socket, visited):
    """Follow only the diffuse color branch. A bump or roughness map is never a substitute for albedo."""
    for link in socket.links:
        node = link.from_node
        if node in visited:
            continue
        visited.add(node)
        if node.type == "TEX_IMAGE" and node.image:
            return node.image
        for source in node.inputs:
            found = diffuse_image(source, visited)
            if found:
                return found
    return None


prepared_images = {}


def preview_image(image):
    if image not in prepared_images:
        # Resolve the path against the linked .blend which owns the image (desks, lamps, etc.).
        path = bpy.path.abspath(image.filepath, library=image.library)
        local = bpy.data.images.load(path, check_existing=False)
        w, h = local.size
        limit = 1024 if "blackboard" in image.name.lower() else 512
        if max(w, h) > limit:
            scale = limit / max(w, h)
            local.scale(max(1, round(w * scale)), max(1, round(h * scale)))
        local.pack()
        prepared_images[image] = local
    return prepared_images[image]


materials = {}


def preview_material(material):
    if material in materials:
        return materials[material]
    out = bpy.data.materials.new("hero-" + material.name)
    out.use_nodes = True
    bsdf = out.node_tree.nodes.get("Principled BSDF")
    nodes = material.node_tree.nodes if material.node_tree else []
    diffuse = next((node for node in nodes if node.type == "BSDF_DIFFUSE"), None)
    color = diffuse.inputs["Color"] if diffuse else None
    base = color.default_value if color and not color.is_linked else material.diffuse_color
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = 0.55
    img = diffuse_image(color, set()) if color else None
    if img:
        node = out.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = preview_image(img)
        out.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    if "metal" in material.name.lower():
        bsdf.inputs["Metallic"].default_value = 0.8
        bsdf.inputs["Roughness"].default_value = 0.32
    # The source has Cycles-only sky portals and blackbody emitters. Keep their locations.
    emission = {"dayLight_portal": (4.0, 4.6, 5.6), "blackBoardLight": (5.0, 4.1, 2.9),
                "ceillingLamp_light": (6.0, 5.2, 3.8)}
    if material.name in emission:
        rgb = emission[material.name]
        bsdf.inputs["Emission Color"].default_value = [c / max(rgb) for c in rgb] + [1]
        bsdf.inputs["Emission Strength"].default_value = max(rgb)
    if any(n.type == "BSDF_GLASS" for n in nodes):
        bsdf.inputs["Transmission Weight"].default_value = 1
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Roughness"].default_value = 0.04
    materials[material] = out
    return out


if kind == "classroom":
    scene = bpy.context.scene
    cam = scene.camera
    rotation = cam.matrix_world.to_quaternion()
    eye = cam.matrix_world.translation
    forward = rotation @ Vector((0, 0, -1))
    # angle_y alone ignores the sensor-fit and render aspect of the authored camera.
    corner = cam.data.view_frame(scene=scene)[0]
    fov = math.degrees(2 * math.atan(abs(corner.y / corner.z)))
    camera = dict(type="perspective", position=web(eye), target=web(eye + forward),
                  up=web(rotation @ Vector((0, 1, 0))), yFovDeg=fov)
    deps = bpy.context.evaluated_depsgraph_get()
    # Cache evaluated meshes; desks and chairs are real collection instances in this file.
    copies, meshes, lights, emitters = [], {}, [], []
    for instance in deps.object_instances:
        obj = instance.object
        if obj.hide_render or any(c.hide_render for c in obj.users_collection):
            continue
        transform = instance.matrix_world.copy()
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            lights.append(dict(type="directional", direction=web(transform.to_quaternion() @ Vector((0, 0, -1))),
                               irradiance=[3.5, 3.1, 2.6], angularRadius=0.02))
        if obj.type == "LIGHT" and obj.data.type == "AREA":
            # Blender area lamps emit along local -Z. Represent the authored fixture by two triangles.
            x = obj.data.size / 2
            y = obj.data.size_y / 2 if obj.data.shape == "RECTANGLE" else x
            corners = [web(transform @ Vector((u, v, 0))) for u, v in [(-x, -y), (-x, y), (x, y), (x, -y)]]
            strength = min(100, obj.data.energy / (4 * x * y * math.pi))
            emitters.append(dict(name=obj.name, positions=sum(corners, []),
                                 emission=[strength * c for c in obj.data.color]))
        if obj.type not in ("MESH", "CURVE", "FONT"):
            continue
        key = obj.original
        if key not in meshes:
            mesh = bpy.data.meshes.new_from_object(obj, preserve_all_data_layers=True, depsgraph=deps)
            for index, slot in enumerate(obj.material_slots):
                if slot.material and index < len(mesh.materials):
                    mesh.materials[index] = preview_material(slot.material)
            meshes[key] = mesh
        copies.append((obj.name, meshes[key], transform))
    clean = bpy.data.scenes.new("WebGPU Classroom")
    bpy.context.window.scene = clean
    for name, mesh, transform in copies:
        obj = bpy.data.objects.new(name, mesh)
        obj.matrix_world = transform
        clean.collection.objects.link(obj)
    (output / "classroom-camera.json").write_text(json.dumps(dict(camera=camera, lights=lights, emitters=emitters)))
elif kind == "sponza":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    folder = assets / "downloads" / "sponza"
    # Keep original glTF material factors, UVs and alpha. This variant ships base-color maps only.
    doc = json.loads((folder / "Sponza.gltf").read_text())
    used = set()
    for mat in doc["materials"]:
        pbr = mat.get("pbrMetallicRoughness", {})
        for key in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            mat.pop(key, None)
        pbr.pop("metallicRoughnessTexture", None)
        if "baseColorTexture" in pbr:
            used.add(pbr["baseColorTexture"]["index"])
    indices = sorted(used)
    images = [doc["images"][doc["textures"][i]["source"]] for i in indices]
    doc["textures"] = [dict(doc["textures"][i], source=k) for k, i in enumerate(indices)]
    doc["images"] = images
    for mat in doc["materials"]:
        tex = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if tex:
            tex["index"] = indices.index(tex["index"])
    slim = folder / "hero-base-color.gltf"
    slim.write_text(json.dumps(doc))
    bpy.ops.import_scene.gltf(filepath=str(slim))
    for img in list(bpy.data.images):
        if img.source == "FILE" and img.size[0]:
            w, h = img.size
            scale = min(1, 512 / max(w, h))
            img.scale(max(1, round(w * scale)), max(1, round(h * scale)))
            img.pack()
else:
    raise ValueError("Expected classroom or sponza")

bpy.ops.export_scene.gltf(filepath=str(output / (kind + ".glb")), export_format="GLB",
                          use_active_scene=True, export_yup=True, export_animations=False,
                          export_cameras=False, export_lights=False)
