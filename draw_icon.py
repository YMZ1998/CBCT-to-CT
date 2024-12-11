import matplotlib.pyplot as plt
from PIL import Image

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Increase the size for a more balanced look

# Set axis limits and hide the axes
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axis('off')  # Hide the axis

ax.text(0, 0.5, 'CBCT', fontsize=120, fontweight='bold', va='center', ha='center', alpha=0.8, color='blue')
ax.text(0, -0.5, '2CT', fontsize=120, fontweight='bold', va='center', ha='center', alpha=0.8, color='blue')

# Save the image with transparent background and no extra padding
plt.savefig('cbct2ct_logo.png', transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)

# Show the image
plt.show()

# Open the PNG image
png_image = Image.open('cbct2ct_logo.png')

# Convert and save as ICO
png_image.save('cbct2ct_logo.ico', format='ICO')
