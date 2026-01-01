# =====================================================================
# Final filtered clinical trigger dictionary 
# (all generic terms, filler phrases and words without a clear local visual anchor removed)
# =====================================================================

# Tier 1: high-risk pathologies and abnormal findings (concrete, localisable lesions)
# Removed: "abnormalities", "abnormality", "disease" (too generic, no specific visual shape)
pathology_triggers = [
    "aspiration", "atelectasis", "atelectatic",
    "bulla", "bullae", "calcification", "calcifications", "cardiomegaly", "chf",
    "collapse", "congestion", "consolidation", "consolidations", "copd",
    "deformity", "edema", "effusion", "effusions", "emphysema",
    "fibrosis", "fracture", "fractures", "granuloma", "hernia", "infection",
    "infectious", "infiltrate", "infiltrates", "infiltration", "lesion",
    "mass", "masses", "nodular", "nodule", "nodules", "opacification",
    "opacities", "opacity", "overload", "pneumonia", "pneumothorax",
    "scarring", "thickening"
]

# Tier 2: key anatomical structures and devices (models often hallucinate their position)
# Removed: "body", "structures", "tissue", "surfaces", "zone" (no localisation anchor; they trigger large-area spurious erasure)
anatomy_device_triggers = [
    "abdomen", "aorta", "aortic", "apex", "arch", "artery", "atrium",
    "base", "bases", "basilar", "bibasilar", "bony", "bronchovascular",
    "cabg", "cardiac", "cardiomediastinal", "cardiomediastinum", "cardiopulmonary",
    "carina", "cath", "catheter", "cavoatrial", "clavicle", "clavicles", "clips",
    "costophrenic", "device", "devices", "diaphragm", "endotracheal", "enteric",
    "heart", "hemidiaphragm", "hemithorax", "hiatal", "hila", "hilar", "hilum",
    "intrathoracic", "jugular", "junction", "lead", "leads", "lobe", "lung", "lungs",
    "markings", "mediastinal", "mediastinum", "nasogastric", "osseous", "pacemaker",
    "pacer", "parenchymal", "picc", "pigtail", "pleura", "pleural", "port",
    "pulmonary", "quadrant", "rib", "ribs", "silhouette", "silhouettes", "spinal",
    "spine", "sternotomy", "stomach", "subclavian", "subcutaneous",
    "surgical", "svc", "thoracic", "trachea", "tracheostomy",
    "tube", "tubes", "valve", "vascular", "vascularity", "vasculature", "vein",
    "venous", "ventricle", "vertebral", "vessel", "vessels", "wires"
]

# Tier 3: frequent radiographic descriptors (strongly visual morphological attributes)
# Removed: "radiographic", "radiographs" (meta-terms about the medium, not visual features)
# Removed: "loss", "dual", "median" (too abstract for 2D SNR localisation on the feature map)
descriptive_triggers = [
    "acute", "adjacent", "air", "airspace", "anterior", "bilaterally", "blunting",
    "borderline", "calcified", "central", "chronic", "crowding", "degenerative",
    "density", "descending", "diffuse", "displaced", "distal",
    "engorged", "engorgement", "enlarged", "enlargement", "fluid", "focal",
    "hyperinflated", "internal", "interstitial", "linear", "loculated",
    "multifocal", "patchy", "perihilar", "posterior", "prominence",
    "prominent", "retrocardiac", "streaky", "tortuosity", "tortuous", "widening"
]