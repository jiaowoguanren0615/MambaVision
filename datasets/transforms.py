DEFAULT_CROP_PCT = 0.875

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)


def resolve_data_config(model, args, default_cfg={}, verbose=True):
    new_config = {}
    default_cfg = default_cfg
    if not default_cfg and model is not None and hasattr(model, 'default_cfg'):
        default_cfg = model.default_cfg

    # Resolve input/image size
    # FIXME grayscale/chans arg to use different # channels?
    in_chans = 3
    input_size = (in_chans, 224, 224)
    if args.img_size is not None:
        # FIXME support passing img_size as tuple, non-square
        assert isinstance(args.img_size, int)
        input_size = (in_chans, args.img_size, args.img_size)
    elif 'input_size' in default_cfg:
        input_size = default_cfg['input_size']
    new_config['input_size'] = input_size

    # resolve interpolation method
    new_config['interpolation'] = 'bicubic'
    if args.interpolation:
        new_config['interpolation'] = args.interpolation
    elif 'interpolation' in default_cfg:
        new_config['interpolation'] = default_cfg['interpolation']

    # resolve datasets + model mean for normalization
    new_config['mean'] = IMAGENET_DEFAULT_MEAN
    if args.mean is not None:
        mean = tuple(args.mean)
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        new_config['mean'] = mean
    elif 'mean' in default_cfg:
        new_config['mean'] = default_cfg['mean']

    # resolve datasets + model std deviation for normalization
    new_config['std'] = IMAGENET_DEFAULT_STD
    if args.std is not None:
        std = tuple(args.std)
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        new_config['std'] = std
    elif 'std' in default_cfg:
        new_config['std'] = default_cfg['std']

    # resolve default crop percentage
    new_config['crop_pct'] = DEFAULT_CROP_PCT
    if args.crop_pct is not None:
        new_config['crop_pct'] = args.crop_pct
    elif 'crop_pct' in default_cfg:
        new_config['crop_pct'] = default_cfg['crop_pct']

    if verbose:
        print('Data processing configuration for current model + datasets:')
        for n, v in new_config.items():
            print('\t%s: %s' % (n, str(v)))

    return new_config