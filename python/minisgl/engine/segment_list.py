from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SegNode:
    start: int
    length: int
    prev: SegNode | None = field(default=None, repr=False)
    next: SegNode | None = field(default=None, repr=False)


class SegmentList:
    """Sorted doubly-linked lists of clean (free) and dirty (allocated) segments."""

    def __init__(self, total_length: int):
        self._clean_head = SegNode(-1, 0)
        self._dirty_head = SegNode(-1, 0)
        if total_length > 0:
            initial = SegNode(0, total_length)
            self._clean_head.next = initial
            initial.prev = self._clean_head

    # ------------------------------------------------------------------ helpers
    def _iter_clean(self):
        node = self._clean_head.next
        while node is not None:
            yield node
            node = node.next

    def _iter_dirty(self):
        node = self._dirty_head.next
        while node is not None:
            yield node
            node = node.next

    def _remove(self, node: SegNode) -> None:
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        node.prev = node.next = None

    def _insert_after(self, anchor: SegNode, node: SegNode) -> None:
        node.next = anchor.next
        node.prev = anchor
        if anchor.next is not None:
            anchor.next.prev = node
        anchor.next = node

    def _insert_clean_sorted(self, node: SegNode) -> None:
        prev = self._clean_head
        nxt = self._clean_head.next
        while nxt is not None and nxt.start < node.start:
            prev = nxt
            nxt = nxt.next
        self._insert_after(prev, node)

    def _insert_dirty_sorted(self, node: SegNode) -> None:
        prev = self._dirty_head
        nxt = self._dirty_head.next
        while nxt is not None and nxt.start < node.start:
            prev = nxt
            nxt = nxt.next
        self._insert_after(prev, node)

    def _move_to_dirty(self, node: SegNode) -> None:
        self._remove(node)
        self._insert_dirty_sorted(node)

    def _move_to_clean(self, node: SegNode) -> None:
        self._remove(node)
        self._insert_clean_sorted(node)
        self._coalesce(node)

    def _coalesce(self, node: SegNode) -> None:
        # merge with prev if adjacent
        if node.prev is not None and node.prev is not self._clean_head:
            if node.prev.start + node.prev.length == node.start:
                prev_node = node.prev
                prev_node.length += node.length
                self._remove(node)
                node = prev_node
        # merge with next if adjacent
        if node.next is not None:
            if node.start + node.length == node.next.start:
                node.length += node.next.length
                self._remove(node.next)

    # ------------------------------------------------------------------ public
    @property
    def total_clean(self) -> int:
        return sum(node.length for node in self._iter_clean())

    @property
    def total_dirty(self) -> int:
        return sum(node.length for node in self._iter_dirty())

    def allocate(self, length: int) -> List[Tuple[int, int]]:
        """Allocate `length` units, possibly from multiple clean segments.
        Returns list of (start, seg_length) ranges moved to dirty.
        """
        if length == 0:
            return []
        remaining = length
        result: List[Tuple[int, int]] = []
        node = self._clean_head.next
        while node is not None and remaining > 0:
            nxt = node.next
            if node.length <= remaining:
                # take whole node
                result.append((node.start, node.length))
                remaining -= node.length
                self._move_to_dirty(node)
            else:
                # split node
                allocated = SegNode(node.start, remaining)
                node.start += remaining
                node.length -= remaining
                result.append((allocated.start, allocated.length))
                self._insert_dirty_sorted(allocated)
                remaining = 0
            node = nxt
        if remaining > 0:
            raise RuntimeError(f"Out of memory: needed {length}, short by {remaining}")
        return result

    def free(self, start: int, length: int) -> None:
        """Free a range [start, start+length) that is currently dirty."""
        if length <= 0:
            return
        end = start + length
        node = self._dirty_head.next
        while node is not None:
            nxt = node.next
            node_end = node.start + node.length
            if node_end <= start:
                node = nxt
                continue
            if node.start >= end:
                break
            # there is overlap
            if node.start >= start and node_end <= end:
                # fully covered
                self._move_to_clean(node)
            elif node.start < start and node_end > end:
                # middle split
                left = SegNode(node.start, start - node.start)
                right = SegNode(end, node_end - end)
                self._remove(node)
                self._insert_dirty_sorted(left)
                self._insert_dirty_sorted(right)
                new_clean = SegNode(start, length)
                self._insert_clean_sorted(new_clean)
                self._coalesce(new_clean)
            elif node.start < start < node_end <= end:
                # overlap at tail
                new_clean = SegNode(start, node_end - start)
                node.length = start - node.start
                self._insert_clean_sorted(new_clean)
                self._coalesce(new_clean)
            elif start <= node.start < end < node_end:
                # overlap at head
                new_clean = SegNode(node.start, end - node.start)
                node.start = end
                node.length = node_end - end
                self._insert_clean_sorted(new_clean)
                self._coalesce(new_clean)
            node = nxt

    def mark_dirty(self, start: int, length: int) -> None:
        """Move a range from clean to dirty."""
        if length <= 0:
            return
        end = start + length
        node = self._clean_head.next
        while node is not None:
            nxt = node.next
            node_end = node.start + node.length
            if node_end <= start:
                node = nxt
                continue
            if node.start >= end:
                break
            if node.start >= start and node_end <= end:
                self._move_to_dirty(node)
            elif node.start < start and node_end > end:
                left = SegNode(node.start, start - node.start)
                right = SegNode(end, node_end - end)
                self._remove(node)
                self._insert_clean_sorted(left)
                self._coalesce(left)
                self._insert_clean_sorted(right)
                self._coalesce(right)
                new_dirty = SegNode(start, length)
                self._insert_dirty_sorted(new_dirty)
            elif node.start < start < node_end <= end:
                new_dirty = SegNode(start, node_end - start)
                node.length = start - node.start
                self._insert_dirty_sorted(new_dirty)
            elif start <= node.start < end < node_end:
                new_dirty = SegNode(node.start, end - node.start)
                node.start = end
                node.length = node_end - end
                self._insert_dirty_sorted(new_dirty)
            node = nxt

    def mark_clean(self, start: int, length: int) -> None:
        """Move a range from dirty to clean."""
        self.free(start, length)

    def find_dirty(self, start: int, length: int) -> List[Tuple[int, int]]:
        """Return sub-ranges of dirty segments overlapping [start, start+length)."""
        end = start + length
        result: List[Tuple[int, int]] = []
        for node in self._iter_dirty():
            node_end = node.start + node.length
            if node_end <= start:
                continue
            if node.start >= end:
                break
            result.append((max(node.start, start), min(node_end, end)))
        return result

    def to_clean_list(self) -> List[Tuple[int, int]]:
        return [(node.start, node.length) for node in self._iter_clean()]

    def to_dirty_list(self) -> List[Tuple[int, int]]:
        return [(node.start, node.length) for node in self._iter_dirty()]
