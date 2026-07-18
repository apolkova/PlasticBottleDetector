# Dataset Description

## Classes and Naming

The dataset uses four object classes:

| Class ID  | Class name    | Meaning                                       |
|---:       |---            |---                                            |
| 0         | bottle        | The visible plastic bottle body               |
| 1         | cap           | The visible bottle cap                        |
| 2         | label         | The visible product label on the bottle       |
| 3         | liquid        | The visible liquid region inside the bottle   |

Class names are written in lowercase because this is the format used in the YOLOv8 dataset.

## How to Interpret Labels

Each label represents a bounding box around one visible object or bottle component.

- `bottle` means the visible body of the plastic bottle.
- `cap` means the cap on top of the bottle, but only when it is visible.
- `label` means the visible product label on the bottle.
- `liquid` means the visible liquid area inside the bottle.

If a part is missing, hidden, or not clearly visible, it is not labeled. For example, if a bottle has no visible cap, there is no `cap` label for that image. If the liquid cannot be clearly seen, the `liquid` class is not annotated.

The goal of these labels is to support a quality-control task. A complete bottle should contain all four classes: bottle, cap, label, and liquid.

## Known Issues and Ambiguous Cases

Some cases were difficult to label consistently:

- Transparent bottles sometimes made the liquid hard to see.
- Dark bottles or dark liquids sometimes made the liquid region unclear.
- Reflections and glare sometimes made labels harder to detect.
- Small caps were difficult to annotate precisely.
- In some images, the cap or label was only partly visible.
- Liquid could sometimes be confused with reflections or the label area.
- Some bottles were partially visible or photographed from difficult angles.
- Background objects sometimes made the scene more cluttered.

The labels were checked manually, but small inconsistencies may still exist, especially for difficult liquid regions, reflective labels, and very small caps.