import struct


class FragmentIndexParser:
    """Read normalized mfra boxes without applying presentation-time edits."""

    @staticmethod
    def parse(mfra: bytes) -> dict:
        """Return native times, byte offsets, and one-based traf/run/sample references per track."""
        if len(mfra) < 8 or mfra[4:8] != b'mfra' or int.from_bytes(mfra[:4], 'big') != len(mfra):
            raise ValueError('Malformed mfra box')
        tracks = {}
        position = 8
        while position < len(mfra):
            if position + 8 > len(mfra):
                raise ValueError('Truncated mfra child header')
            size, kind = struct.unpack_from('>I4s', mfra, position)
            header_size = 8
            if size == 1:
                if position + 16 > len(mfra):
                    raise ValueError('Truncated extended mfra child header')
                size = struct.unpack_from('>Q', mfra, position + 8)[0]
                header_size = 16
            if size < header_size or position + size > len(mfra):
                raise ValueError('Invalid mfra child size')
            if kind == b'tfra':
                track_id, entries = FragmentIndexParser._parse_tfra(
                    mfra[position + header_size:position + size])
                if track_id in tracks:
                    raise ValueError(f'Duplicate tfra for track {track_id}')
                tracks[track_id] = entries
            position += size
        if not tracks:
            raise ValueError('mfra contains no tfra index')
        return tracks

    @staticmethod
    def _parse_tfra(data: bytes) -> tuple:
        if len(data) < 16:
            raise ValueError('Truncated tfra header')
        version_flags, track_id, lengths, count = struct.unpack_from('>IIII', data)
        version = version_flags >> 24
        if version not in (0, 1) or version_flags & 0xFFFFFF or lengths >> 6:
            raise ValueError('Unsupported tfra version, flags, or reserved fields')
        widths = [((lengths >> shift) & 3) + 1 for shift in (4, 2, 0)]
        time_size = 16 if version else 8
        entry_size = time_size + sum(widths)
        if not count or len(data) != 16 + count * entry_size:
            raise ValueError(f'Empty or truncated tfra entries for track {track_id}')
        entries = []
        position = 16
        for entry_number in range(count):
            time, offset = struct.unpack_from('>QQ' if version else '>II', data, position)
            position += time_size
            reference = []
            for width in widths:
                reference.append(int.from_bytes(data[position:position + width], 'big'))
                position += width
            entries.append((time, offset, *reference))
        return track_id, entries