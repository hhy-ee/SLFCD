import rpack
# Create a bunch of rectangles (width, height)
sizes = [(256, 256), (256, 256), (384, 512), (256, 384), (256, 256)]

# Pack
positions = rpack.pack(sizes, max_height=800, max_width=800)

bbox_size = rpack.bbox_size(sizes, positions)

a = 1