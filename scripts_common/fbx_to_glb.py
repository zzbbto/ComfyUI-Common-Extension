"""
FBX to GLB converter node for ComfyUI-Common-Extension.
Uses Blender (bpy) in a SUBPROCESS.
"""

import os
import subprocess
import tempfile
import logging

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except Exception:
    COMFYUI_OUTPUT_FOLDER = None

log = logging.getLogger("comfyui_common")


class ConvertFBXToGLB:
    """
    Convert rigged FBX to GLB via Blender subprocess.
    glb_path supports relative paths (based on ComfyUI output folder).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path": ("STRING", {"default": "", "multiline": False}),
                "glb_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_path", "glb_path_relative")
    FUNCTION = "execute"
    CATEGORY = "Common/3D"

    def execute(self, fbx_path, glb_path):
        # ── 支持相对路径：优先 output，fallback input ──
        if fbx_path and not os.path.isabs(fbx_path):
            if COMFYUI_OUTPUT_FOLDER:
                candidate = os.path.join(COMFYUI_OUTPUT_FOLDER, fbx_path)
                if os.path.exists(candidate):
                    fbx_path = candidate

            if not os.path.exists(fbx_path):
                try:
                    import folder_paths
                    input_folder = folder_paths.get_input_directory()
                    candidate = os.path.join(input_folder, fbx_path)
                    if os.path.exists(candidate):
                        fbx_path = candidate
                except Exception:
                    pass

        if not fbx_path or not os.path.exists(fbx_path):
            raise ValueError(f"FBX file not found: {fbx_path}")

        fbx_path = os.path.abspath(fbx_path)
        fbx_name = os.path.splitext(os.path.basename(fbx_path))[0]

        # 解析 glb_path：支持相对路径、绝对路径、纯目录
        if not glb_path:
            # 情况1：空字符串，自动用 FBX 同名
            glb_path = fbx_path.rsplit('.', 1)[0] + '.glb'
        elif os.path.isabs(glb_path):
            # 情况2：绝对路径，直接用
            if glb_path.endswith(('/', '\\')):
                glb_path = os.path.join(glb_path, fbx_name + '.glb')
        else:
            # 情况3：相对路径，基于 ComfyUI output 目录拼接
            if COMFYUI_OUTPUT_FOLDER:
                glb_path = os.path.join(COMFYUI_OUTPUT_FOLDER, glb_path)
            else:
                glb_path = os.path.join(os.getcwd(), glb_path)

            # 如果只填了目录（以 / 或 \ 结尾），自动补全文件名
            if glb_path.endswith(('/', '\\')) or os.path.isdir(glb_path):
                glb_path = os.path.join(glb_path, fbx_name + '.glb')
            # 如果没有 .glb 后缀，自动补上
            elif not glb_path.lower().endswith('.glb'):
                glb_path += '.glb'

        glb_path = os.path.abspath(glb_path)
        out_dir = os.path.dirname(glb_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # 计算相对路径（相对于 ComfyUI output 目录）
        glb_relative = glb_path
        if COMFYUI_OUTPUT_FOLDER and glb_path.startswith(os.path.abspath(COMFYUI_OUTPUT_FOLDER)):
            glb_relative = os.path.relpath(glb_path, COMFYUI_OUTPUT_FOLDER)
            glb_relative = glb_relative.replace("\\", "/")

        # 删除旧 GLB，避免失败后返回旧文件
        if os.path.exists(glb_path):
            try:
                os.remove(glb_path)
                log.info("Removed old GLB: %s", glb_path)
            except Exception as e:
                log.warning("Failed to remove old GLB: %s", e)

        # Build Blender Python script
        script_lines = [
            "import bpy",
            "",
            f"fbx = r'{fbx_path}'",
            f"glb = r'{glb_path}'",
            "",
            "bpy.ops.wm.read_factory_settings(use_empty=True)",
            "bpy.ops.object.select_all(action='SELECT')",
            "bpy.ops.object.delete(use_global=False)",
            "",
            "bpy.ops.import_scene.fbx(filepath=fbx)",
            "",
            "bpy.ops.export_scene.gltf(",
            "    filepath=glb,",
            "    export_format='GLB',",
            "    export_yup=True,",
            "    export_apply=True,",
            "    export_texcoords=True,",
            "    export_normals=True,",
            "    export_materials='EXPORT',",
            "    export_image_format='AUTO',",
            ")",
            "",
            "print('GLB export complete:', glb)",
        ]
        script_content = "\n".join(script_lines)

        script_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(script_content)
                script_file = f.name

            log.info("Converting FBX -> GLB...")
            log.info("  FBX: %s", fbx_path)
            log.info("  GLB: %s", glb_path)

            result = subprocess.run(
                [subprocess.sys.executable, script_file],
                capture_output=True,
                text=True,
                timeout=180,
            )

            if not os.path.exists(glb_path):
                err = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"Blender conversion failed (no output file):\n{err}")

            file_size = os.path.getsize(glb_path)
            if file_size == 0:
                raise RuntimeError("Blender created an empty GLB file.")

            if result.returncode != 0 and result.stderr.strip():
                log.warning("Blender warnings (exit %d):\n%s",
                           result.returncode, result.stderr.strip())

            log.info("Successfully converted to GLB: %s (%d bytes)", glb_path, file_size)
            return (glb_path, glb_relative)

        finally:
            if script_file and os.path.exists(script_file):
                os.remove(script_file)


NODE_CLASS_MAPPINGS = {
    "ConvertFBXToGLB": ConvertFBXToGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertFBXToGLB": "Common: FBX to GLB",
}