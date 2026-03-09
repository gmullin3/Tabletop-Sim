features = {

    "reward": {
        "dtype": "float64",
        "shape": (1,),
    }, 

    "done": {
        "dtype": "bool",
        "shape": (1,),    
    },



    "observation.state.joint_pos": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "motors": [
                "l1",
                "l2",
                "l3",
                "l4",
                "l5",
                "l6",
                "l_gripper",
                "r1",
                "r2",
                "r3",
                "r4",
                "r5",
                "r6",
                "r_gripper",
            ]
        },
    },

    "observation.state.ee_6d_pos": {
        "dtype": "float32",
        "shape": (20,),
        "names": {
            "motors": [
                "lx",
                "ly",
                "lz",
                "l_c1x",
                "l_c1y",
                "l_c1z",
                "l_c2x",
                "l_c2y",
                "l_c2z",
                "l_gripper",
                "rx",
                "ry",
                "rz",
                "r_c1x",
                "r_c1y",
                "r_c1z",
                "r_c2x",
                "r_c2y",
                "r_c2z",
                "r_gripper",
            ]
        },
    },

    "observation.state.ee_quat_pos": {
        "dtype": "float32",
        "shape": (16,),
        "names": {
            "motors": [
                "lx",
                "ly",
                "lz",
                "l_qx",
                "l_qy",
                "l_qz",
                "l_qw",
                "l_gripper",
                "rx",
                "ry",
                "rz",
                "r_qx",
                "r_qy",
                "r_qz",
                "r_qw",
                "r_gripper",
            ]
        },
    },
    "observation.images.agentview": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 480,
            "video.width": 640,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "observation.images.wrist_left": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 240,
            "video.width": 320,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "observation.images.wrist_right": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 240,
            "video.width": 320,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "action.joint_pos": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "motors": [
                "l1",
                "l2",
                "l3",
                "l4",
                "l5",
                "l6",
                "l_gripper",
                "r1",
                "r2",
                "r3",
                "r4",
                "r5",
                "r6",
                "r_gripper",
            ]
        },
   },

    "action.ee_6d_pos": {
        "dtype": "float32",
        "shape": (20,),
        "names": {
            "motors": [
                "lx",
                "ly",
                "lz",
                "l_c1x",
                "l_c1y",
                "l_c1z",
                "l_c2x",
                "l_c2y",
                "l_c2z",
                "l_gripper",
                "rx",
                "ry",
                "rz",
                "r_c1x",
                "r_c1y",
                "r_c1z",
                "r_c2x",
                "r_c2y",
                "r_c2z",
                "r_gripper",
            ]
        },
    },

    "action.ee_quat_pos": {
        "dtype": "float32",
        "shape": (16,),
        "names": {
            "motors": [
                "lx",
                "ly",
                "lz",
                "l_qx",
                "l_qy",
                "l_qz",
                "l_qw",
                "l_gripper",
                "rx",
                "ry",
                "rz",
                "r_qx",
                "r_qy",
                "r_qz",
                "r_qw",
                "r_gripper",
            ]
        },
    },
}



anubis_features_large = {

    "reward": {
        "dtype": "float64",
        "shape": (1,),
    }, 

    "done": {
        "dtype": "bool",
        "shape": (1,),    
    },

    "observation.state.ee_6d_pos": {
        "dtype": "float32",
        "shape": (20,),
        "names": {
            "motors": [
                "lx",
                "ly",
                "lz",
                "l_c1x",
                "l_c1y",
                "l_c1z",
                "l_c2x",
                "l_c2y",
                "l_c2z",
                "l_gripper",
                "rx",
                "ry",
                "rz",
                "r_c1x",
                "r_c1y",
                "r_c1z",
                "r_c2x",
                "r_c2y",
                "r_c2z",
                "r_gripper",
            ]
        },
    },
    "observation.images.agentview": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 480,
            "video.width": 640,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "observation.images.wrist_left": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 480,
            "video.width": 640,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "observation.images.wrist_right": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 480,
            "video.width": 640,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "action.ee_6d_pos": {
        "dtype": "float32",
        "shape": (20,),
        "names": {
            "motors": [
                "lx",
                "ly",
                "lz",
                "l_c1x",
                "l_c1y",
                "l_c1z",
                "l_c2x",
                "l_c2y",
                "l_c2z",
                "l_gripper",
                "rx",
                "ry",
                "rz",
                "r_c1x",
                "r_c1y",
                "r_c1z",
                "r_c2x",
                "r_c2y",
                "r_c2z",
                "r_gripper",
            ]
        },
    },
}

anubis_features_small = {

    "reward": {
        "dtype": "float64",
        "shape": (1,),
    }, 

    "done": {
        "dtype": "bool",
        "shape": (1,),    
    },

    "observation.state.ee_6d_pos": {
        "dtype": "float32",
        "shape": (20,),
        "names": {
            "motors": [
                "lx",
                "ly",
                "lz",
                "l_c1x",
                "l_c1y",
                "l_c1z",
                "l_c2x",
                "l_c2y",
                "l_c2z",
                "l_gripper",
                "rx",
                "ry",
                "rz",
                "r_c1x",
                "r_c1y",
                "r_c1z",
                "r_c2x",
                "r_c2y",
                "r_c2z",
                "r_gripper",
            ]
        },
    },
    "observation.images.agentview": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 240,
            "video.width": 320,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "observation.images.wrist_left": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 240,
            "video.width": 320,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "observation.images.wrist_right": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": 20.0,
            "video.height": 240,
            "video.width": 320,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    },

    "action.ee_6d_pos": {
        "dtype": "float32",
        "shape": (20,),
        "names": {
            "motors": [
                "lx",
                "ly",
                "lz",
                "l_c1x",
                "l_c1y",
                "l_c1z",
                "l_c2x",
                "l_c2y",
                "l_c2z",
                "l_gripper",
                "rx",
                "ry",
                "rz",
                "r_c1x",
                "r_c1y",
                "r_c1z",
                "r_c2x",
                "r_c2y",
                "r_c2z",
                "r_gripper",
            ]
        },
    },
}