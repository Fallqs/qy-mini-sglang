from minisgl.engine.segment_list import SegmentList


def test_initial_state():
    sl = SegmentList(100)
    assert sl.total_clean == 100
    assert sl.total_dirty == 0
    assert sl.to_clean_list() == [(0, 100)]


def test_allocate_exact():
    sl = SegmentList(100)
    segs = sl.allocate(100)
    assert segs == [(0, 100)]
    assert sl.total_clean == 0
    assert sl.to_dirty_list() == [(0, 100)]


def test_allocate_partial():
    sl = SegmentList(100)
    segs = sl.allocate(30)
    assert segs == [(0, 30)]
    assert sl.total_clean == 70
    assert sl.to_clean_list() == [(30, 70)]
    assert sl.to_dirty_list() == [(0, 30)]


def test_allocate_then_free():
    sl = SegmentList(100)
    sl.allocate(30)
    sl.free(0, 30)
    assert sl.to_clean_list() == [(0, 100)]
    assert sl.total_dirty == 0


def test_allocate_non_contiguous():
    sl = SegmentList(100)
    sl.allocate(20)
    sl.allocate(20)
    sl.free(0, 20)  # free first block
    segs = sl.allocate(30)
    assert segs == [(0, 20), (40, 10)]
    assert sl.to_dirty_list() == [(0, 20), (20, 20), (40, 10)]


def test_free_coalesce():
    sl = SegmentList(100)
    sl.allocate(20)
    sl.allocate(20)
    sl.allocate(20)
    # dirty: 0-20, 20-40, 40-60
    sl.free(20, 20)
    sl.free(0, 20)
    # should coalesce to 0-40; 40-60 remains dirty
    assert sl.to_clean_list() == [(0, 40), (60, 40)]
    assert sl.to_dirty_list() == [(40, 20)]


def test_mark_dirty_clean():
    sl = SegmentList(100)
    sl.mark_dirty(10, 20)
    assert sl.to_clean_list() == [(0, 10), (30, 70)]
    assert sl.to_dirty_list() == [(10, 20)]
    sl.mark_clean(10, 20)
    assert sl.to_clean_list() == [(0, 100)]


def test_mark_dirty_middle_split():
    sl = SegmentList(100)
    sl.mark_dirty(10, 5)
    assert sl.to_clean_list() == [(0, 10), (15, 85)]
    assert sl.to_dirty_list() == [(10, 5)]


def test_find_dirty():
    sl = SegmentList(100)
    sl.allocate(20)
    sl.allocate(20)
    # find_dirty returns (overlap_start, overlap_end)
    assert sl.find_dirty(10, 15) == [(10, 20), (20, 25)]
    assert sl.find_dirty(15, 10) == [(15, 20), (20, 25)]
    assert sl.find_dirty(0, 50) == [(0, 20), (20, 40)]


def test_free_partial_overlap_tail():
    sl = SegmentList(100)
    sl.mark_dirty(0, 20)
    sl.free(10, 20)
    assert sl.to_dirty_list() == [(0, 10)]
    assert sl.to_clean_list() == [(10, 90)]


def test_free_partial_overlap_head():
    sl = SegmentList(100)
    sl.mark_dirty(10, 20)
    sl.free(0, 15)
    assert sl.to_dirty_list() == [(15, 15)]
    assert sl.to_clean_list() == [(0, 15), (30, 70)]
