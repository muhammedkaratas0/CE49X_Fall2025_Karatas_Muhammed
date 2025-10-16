# How to Add Your Own Images to the HTML Code

## Method 1: Using Local Image Files (RECOMMENDED)

### Step 1: Save Your Images
Save your images in the `Week_4` folder with these names:
- `cosmo_logo.png` - Your Cosmo logo
- `spiral_icon.png` - Your blue spiral icon
- `profile_photo.png` - Profile picture (optional)

### Step 2: Use in HTML
The HTML code already has placeholders for these images. Just save your images with the correct names!

```html
<!-- Logo Example -->
<img src="cosmo_logo.png" alt="Cosmo Logo" style="height: 50px;">

<!-- Icon Example -->
<img src="spiral_icon.png" alt="Study Icon" style="width: 52px; height: 52px;">
```

---

## Method 2: Using Full File Paths

If your images are in a different folder:

```html
<img src="/Users/alikaratas/Desktop/CE49X/Week_4/images/my_logo.png" alt="Logo">
```

---

## Method 3: Using Base64 (Embedded Images)

If you want images embedded directly in HTML (no external files):

### Step 1: Convert Image to Base64
Run this command in Terminal:
```bash
base64 -i your_image.png | pbcopy
```
This copies the base64 code to your clipboard.

### Step 2: Use in HTML
```html
<img src="data:image/png;base64,PASTE_BASE64_HERE" alt="Image">
```

---

## Method 4: Using Online Images (URLs)

If your image is hosted online:
```html
<img src="https://example.com/your-image.png" alt="Image">
```

---

## Quick Setup Instructions

### To Add Your Cosmo Logo:
1. Save your Cosmo logo as: `Week_4/cosmo_logo.png`
2. The HTML will automatically use it!

### To Add Your Spiral Icon:
1. Save your spiral icon as: `Week_4/spiral_icon.png`
2. The HTML will automatically use it!

### To Add Profile Photo:
1. Save your photo as: `Week_4/profile_photo.png`
2. The HTML will automatically use it!

---

## File Formats Supported
- PNG (recommended for logos and icons)
- JPG/JPEG (good for photos)
- SVG (best for scalable graphics)
- GIF (for animations)
- WebP (modern, smaller file size)

---

## Pro Tips
✅ Use PNG for logos with transparency
✅ Use JPG for photographs
✅ Keep image sizes reasonable (< 500KB for web)
✅ Use descriptive file names (no spaces!)
✅ Optimize images before using (compress them)

---

## Need Help?
Just save your images in the Week_4 folder and I'll update the HTML code to use them!
