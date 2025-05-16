from PIL import Image

def combine_images(image_files, dir):
    # Open the images
    images = [Image.open(x) for x in image_files]

    # Calculate maximum width and height of images
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # New image size should be two images wide by two images tall
    new_image = Image.new('RGB', (max_width * 2, max_height * 2))

    # Calculate positions to center each image in its grid cell
    positions = [
        ((max_width - images[0].width) // 2, (max_height - images[0].height) // 2),
        (max_width + (max_width - images[1].width) // 2, (max_height - images[1].height) // 2),
        ((max_width - images[2].width) // 2, max_height + (max_height - images[2].height) // 2),
        (max_width + (max_width - images[3].width) // 2, max_height + (max_height - images[3].height) // 2)
    ]

    # Paste images into the new image
    for img, pos in zip(images, positions):
        new_image.paste(img, pos)

    new_image.save(f'{dir}acc_chart_l38bchat.png')

def create_image_grid(image_paths, output_path, grid_size=(2, 2)):
    # Load images
    images = [Image.open(path) for path in image_paths]

    # Calculate maximum width and height of images
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new image with a white background
    new_image = Image.new('RGB', (max_width * grid_size[0], max_height * grid_size[1]), 'white')

    # Calculate positions to center each image in its grid cell
    positions = []
    for i in range(len(images)):
        x_pos = (i % grid_size[0]) * max_width + (max_width - images[i].width) // 2
        y_pos = (i // grid_size[0]) * max_height + (max_height - images[i].height) // 2
        positions.append((x_pos, y_pos))

    # Paste images into the new image
    for img, pos in zip(images, positions):
        new_image.paste(img, pos)

    # Save the new image
    new_image.save(dir+output_path)


from PIL import Image

def create_flexible_grid(image_paths, output_path, grid_dims=(2, 2)):
    # Load images
    images = [Image.open(path) for path in image_paths]

    # Determine the dimensions of each row and column
    row_heights = [0] * grid_dims[1]  # Initialize row heights
    col_widths = [0] * grid_dims[0]  # Initialize column widths

    # Assign images to grid positions
    grid = {}
    for idx, img in enumerate(images):
        row = idx // grid_dims[0]
        col = idx % grid_dims[0]
        grid[(row, col)] = img
        # Update the maximum dimensions for each row and column
        row_heights[row] = max(row_heights[row], img.height)
        col_widths[col] = max(col_widths[col], img.width)

    # Create new image with a white background
    total_width = sum(col_widths)
    total_height = sum(row_heights)
    new_image = Image.new('RGB', (total_width, total_height), 'white')

    # Calculate starting positions for each row and column
    y_offsets = [sum(row_heights[:i]) for i in range(grid_dims[1])]
    x_offsets = [sum(col_widths[:i]) for i in range(grid_dims[0])]

    # Paste images into the new image, centering in each cell
    for (row, col), img in grid.items():
        x_pos = x_offsets[col] + (col_widths[col] - img.width) // 2
        y_pos = y_offsets[row] + (row_heights[row] - img.height) // 2
        new_image.paste(img, (x_pos, y_pos))

    # Save the new image
    new_image.save(dir+output_path)


dir = "./"
image_files = [f'{dir}nlls_ft_lora_0to5_learn_interleaved_mixedlmsys4_orthrandembedembed_mult1_400.png',
               f'{dir}nlls_ft_lora_5to10_learn_interleaved_mixedlmsys4_randomlikeembed_mult1_800.png',
               f'{dir}nlls_ft_lora_5to10_learn_interleaved_mixedlmsys4_randomuaembed_mult1_500.png',
               f'{dir}nlls_ft_lora_0to5_interleaved_mixedlmsys4_orthrandembedembed_mult0.2_700.png',
               f'{dir}nlls_ft_lora_5to10_interleaved_mixedlmsys4_orthrandembedembed_mult0.4_300.png',
               f'{dir}nlls_ft_loraattn_5to10_interleaved_mixedlmsys4_orthrandembedembed_mult0.2_300.png',
               f'{dir}nlls_ft_lora_0to5_interleaved_mixedref2_orthrandembedembed_mult0.8_800.png',

               f'{dir}nlls_ft_lora_5to10_interleaved_mixedlmsys4_orthrand00_mult0.4_600.png',
               f'{dir}nlls_ft_lora_0to5_interleaved_mixedlmsys4_orthrand00_mult0.4_300_new_tmp.png',

               f'{dir}nlls_ft_lora_5to10_interleaved_mixedlmsys4_orthrand4_mult0.4_300.png',
               f'{dir}nlls_ft_lora_5to10_interleaved_mixedlmsys4_orthrand4_mult0.4_500.png',
               f'{dir}nlls_ft_lora_5to10_interleaved_mixedlmsys4_orthrand4_mult0.4_700.png',
               f'{dir}nlls_ft_lora_5to10_interleaved_mixedlmsys4_orthrand4_mult0.4_1800.png',
               f'{dir}nlls_ft_lora_10to15_interleaved_mixedlmsys4_orthrand4_mult0.4_300.png',
               f'{dir}nlls_ft_lora_10to15_interleaved_mixedlmsys4_orthrand44b_mult0.4_800.png',
               f'{dir}nlls_ft_lora_10to20_interleaved_mixedlmsys4_orthrand44_mult0.8_700.png',
               f'{dir}nlls_ft_loraattn_10to15_interleaved_mixedlmsys4_orthrand44_mult0.4_700.png',
               f'{dir}nlls_ft_lora_10to15_interleaved_mixedref2_orthrand44_mult0.4_900.png',
               f'{dir}nlls_ft_lora_5to10_interleaved_mixedref3_orthrand44_mult0.4_400.png',
               f'{dir}nlls_ft_lora_10to15_interleaved_mixedref3_orthrand44_mult0.4_600_new_tmp.png']

image_files = [f'{dir}nlls_narafa_harmful_responses_test.png',
               f'{dir}nlls_narafa_harmful_lat_responses_test.png',
               f'{dir}nlls_narafa_harmful3_responses.png',
               f'{dir}nlls_narafa_mean_responses_test.png']
output_file = 'nlls_narafa_all.png'

create_flexible_grid(image_files, output_file, (4,1))

