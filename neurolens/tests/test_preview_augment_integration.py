import math
import numpy as np
import pytest
import tensorflow as tf

from neurolens.src.neurolens.SplitType import SplitType
from neurolens.src.neurolens.TumorClass import TumorClass
from neurolens.src.neurolens.controller.AugmentationConfig import AugmentationConfig
from neurolens.src.neurolens.controller.PreprocessingPreviewer import PreprocessingPreviewer
from neurolens.src.neurolens.data_management.DatasetSplit import DatasetSplit
from neurolens.src.neurolens.data_management.ImageRecord import ImageRecord
from neurolens.src.neurolens.preprocessing.AugmentedBatch import AugmentedBatch
from neurolens.src.neurolens.preprocessing.AugmentedBatchGenerator import AugmentedBatchGenerator
from neurolens.src.neurolens.preprocessing.ImagePreprocessor import ImagePreprocessor
from neurolens.src.neurolens.results.PreprocessingPreview import PreprocessingPreview

SIZE = 32
PER_CLASS = 5
IMAGES_PER_CLASS = 2
COPIES = 2
N_CLASSES = 4


def _write_image(path, h, w, seed, jpg=False):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 1), dtype=np.uint8)
    t = tf.constant(arr)
    data = tf.io.encode_jpeg(t) if jpg else tf.io.encode_png(t)
    tf.io.write_file(str(path), data)


@pytest.fixture
def records(tmp_path):
    out = []
    seed = 0
    for c in TumorClass:
        for i in range(PER_CLASS):
            jpg = (len(out) == 0)
            p = tmp_path / f"{c.name}_{i}.{'jpg' if jpg else 'png'}"
            _write_image(p, h=40 + i, w=30 + 2 * i, seed=seed, jpg=jpg)
            seed += 1
            out.append(ImageRecord(str(p), c, SplitType.TRAINING))
    return out


@pytest.fixture
def split(records):
    s = DatasetSplit()
    s._trainingSamples = list(records)
    return s


@pytest.fixture
def preprocessor(records):
    p = ImagePreprocessor(width=SIZE, height=SIZE)
    p.fitNormalization(records)
    return p


@pytest.fixture
def config():
    return AugmentationConfig(imagesPerClass=IMAGES_PER_CLASS, copiesPerImage=COPIES, seed=7)


def test_decode_shape_dtype_range(preprocessor, records):
    for r in (records[0], records[1]):
        img = preprocessor.decode(r.imagePath)
        assert img.shape == (SIZE, SIZE, 1)
        assert img.dtype == tf.float32
        assert 0.0 <= float(tf.reduce_min(img)) and float(tf.reduce_max(img)) <= 1.0


def test_preprocess_keeps_3d_shape_and_finite(preprocessor, records):
    img = preprocessor.preprocess(records[0].imagePath)
    assert img.shape == (SIZE, SIZE, 1)
    assert img.dtype == tf.float32
    assert bool(tf.reduce_all(tf.math.is_finite(img)))


def test_normalisation_is_standardised(preprocessor, records):
    imgs = tf.stack([preprocessor.preprocess(r.imagePath) for r in records])
    assert abs(float(tf.reduce_mean(imgs))) < 0.2
    assert 0.8 < float(tf.math.reduce_std(imgs)) < 1.2


def test_augment_copies(preprocessor, records):
    img = preprocessor.preprocess(records[0].imagePath)
    out = preprocessor.augment(img, 3)
    assert len(out) == 3
    assert all(a.shape == img.shape for a in out)
    assert preprocessor.augment(img, 0) == []


def test_build_dataset(preprocessor, records):
    ds = preprocessor.buildDataset(records)
    items = list(ds)
    assert len(items) == len(records)
    x, y = items[0]
    assert x.shape == (SIZE, SIZE, 1)
    assert y.shape == (N_CLASSES,)
    assert float(tf.reduce_sum(y)) == 1.0


def test_augmented_batch_add_counts_dataset_release(preprocessor, records):
    batch = AugmentedBatch()
    chosen = records[:3]
    for r in chosen:
        img = preprocessor.preprocess(r.imagePath)
        idx = batch.add(r, img)
        for a in preprocessor.augment(img, COPIES):
            batch.addAugmented(a, idx)
    assert batch.size() == 3
    assert batch.classCounts()[TumorClass.GLIOMA] == 3
    assert batch.classCounts()[TumorClass.PITUITARY] == 0
    assert len(batch.augmentedFor(1)) == COPIES
    assert batch.recordAt(2) is chosen[2]
    ds = batch.toDataset()
    assert len(list(ds)) == 3 * (1 + COPIES)
    assert not batch.isReleased()
    batch.release()
    assert batch.isReleased()
    assert batch.size() == 0


def test_augmented_batch_without_augmented_copies(preprocessor, records):
    batch = AugmentedBatch()
    batch.add(records[0], preprocessor.preprocess(records[0].imagePath))
    assert len(list(batch.toDataset())) == 1


def test_generator_batches_covers_once(preprocessor, records, config):
    gen = AugmentedBatchGenerator(config, preprocessor)
    gen.reset(records)

    seen = []
    n_batches = 0
    while gen.hasNext():
        batch = gen.nextBatch()
        n_batches += 1
        assert all(v <= IMAGES_PER_CLASS for v in batch.classCounts().values())
        seen += [batch.recordAt(i).imagePath for i in range(batch.size())]
        assert len(batch._augmented) == batch.size() * COPIES

    assert n_batches == math.ceil(PER_CLASS / IMAGES_PER_CLASS)
    assert sorted(seen) == sorted(r.imagePath for r in records)
    with pytest.raises(StopIteration):
        gen.nextBatch()


def test_generator_releases_previous_batch(preprocessor, records, config):
    gen = AugmentedBatchGenerator(config, preprocessor)
    gen.reset(records)
    first = gen.nextBatch()
    assert not first.isReleased()
    second = gen.nextBatch()
    assert first.isReleased()
    assert not second.isReleased()


def test_generator_same_seed_same_order(preprocessor, records):
    def first_paths(seed):
        g = AugmentedBatchGenerator(AugmentationConfigFactory(seed), preprocessor)
        g.reset(records)
        b = g.nextBatch()
        return [b.recordAt(i).imagePath for i in range(b.size())]

    assert first_paths(1) == first_paths(1)


def AugmentationConfigFactory(seed):
    return AugmentationConfig(IMAGES_PER_CLASS, COPIES, seed)


def test_generator_as_dataset(preprocessor, records, config):
    gen = AugmentedBatchGenerator(config, preprocessor)
    gen.reset(records)
    ds = gen.asDataset()
    assert ds.element_spec[0].shape == (SIZE, SIZE, 1)
    assert ds.element_spec[1].shape == (N_CLASSES,)
    items = list(ds)
    assert len(items) == len(records) * (1 + COPIES)
    assert all(float(tf.reduce_sum(y)) == 1.0 for _, y in items)
    assert len(list(ds)) == len(items)


def test_generator_zero_copies(preprocessor, records):
    cfg = AugmentationConfig(IMAGES_PER_CLASS, 0, 1)
    gen = AugmentedBatchGenerator(cfg, preprocessor)
    gen.reset(records)
    assert len(list(gen.asDataset())) == len(records)


def test_previewer_first_preview(preprocessor, split, config):
    previewer = PreprocessingPreviewer(preprocessor)
    preview = previewer.start(split, config)

    assert isinstance(preview, PreprocessingPreview)
    assert preview is previewer.current
    assert preview._batchIndex == 0
    assert preview.totalCounts() == IMAGES_PER_CLASS * N_CLASSES
    assert preview.augumentedCounts() == preview.totalCounts() * COPIES

    for c in TumorClass:
        samples = preview.getSamples(c)
        assert len(samples) == IMAGES_PER_CLASS
        for s in samples:
            assert s._label == c
            assert s._original.shape == (SIZE, SIZE, 1)
            assert s._preprocessed.shape == (SIZE, SIZE, 1)
            assert len(s._augmented) == COPIES
            assert all(a.shape == (SIZE, SIZE, 1) for a in s._augmented)


def test_previewer_original_is_unnormalized_preprocessed_is_not(preprocessor, split, config):
    preview = PreprocessingPreviewer(preprocessor).start(split, config)
    s = preview.getSamples(TumorClass.GLIOMA)[0]
    assert 0.0 <= float(tf.reduce_min(s._original)) and float(tf.reduce_max(s._original)) <= 1.0
    assert not np.allclose(s._original.numpy(), s._preprocessed.numpy())


def test_previewer_iterates_all_batches_then_none(preprocessor, split, config):
    previewer = PreprocessingPreviewer(preprocessor)
    first = previewer.start(split, config)
    previews = [first]
    while True:
        p = previewer.next()
        if p is None:
            break
        previews.append(p)
    assert len(previews) == math.ceil(PER_CLASS / IMAGES_PER_CLASS)
    assert [p._batchIndex for p in previews] == list(range(len(previews)))
    assert previews[-1].totalCounts() == (PER_CLASS % IMAGES_PER_CLASS or IMAGES_PER_CLASS) * N_CLASSES
    assert previewer.next() is None


def test_previewer_releases_previous_preview(preprocessor, split, config):
    previewer = PreprocessingPreviewer(preprocessor)
    first = previewer.start(split, config)
    assert first.totalCounts() > 0
    second = previewer.next()
    assert first.totalCounts() == 0
    assert second.totalCounts() > 0


def test_previewer_restart_resets_index(preprocessor, split, config):
    previewer = PreprocessingPreviewer(preprocessor)
    previewer.start(split, config)
    previewer.next()
    again = previewer.start(split, config)
    assert again._batchIndex == 0


def test_previewer_every_training_image_shown_once(preprocessor, split, config):
    previewer = PreprocessingPreviewer(preprocessor)
    paths = []
    p = previewer.start(split, config)
    while p is not None:
        paths += [s._imagePath for c in TumorClass for s in p.getSamples(c)]
        p = previewer.next()
    assert sorted(paths) == sorted(r.imagePath for r in split.getTrainingSet())