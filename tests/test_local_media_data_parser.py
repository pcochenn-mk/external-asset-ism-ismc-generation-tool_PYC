"""
Unit tests for ISO/IEC 14496-12 box headers, edit-list presentation timing,
and standard versus packager-aligned ISMC chunk serialization.

These tests build small synthetic MP4-like files byte by byte instead of using
the JSON test data fixtures used by the integration tests: the JSON fixtures
capture already-extracted moov data (post box-scanning), so they cannot
exercise the raw top-level box scanning logic that these tests target.
"""
import struct
import xml.etree.ElementTree as ET
from fractions import Fraction
from types import SimpleNamespace

import pytest

from tools.pymp4.src.pymp4.parser import Box
from external_asset_ism_ismc_generation_tool.local_file_client.local_file_service_client import LocalFileServiceClient
from external_asset_ism_ismc_generation_tool.media_data_parser.local_media_data_parser import LocalMediaDataParser
from external_asset_ism_ismc_generation_tool.media_data_parser.media_file_data_reader import MediaFileDataReader
from external_asset_ism_ismc_generation_tool.media_data_parser.model.atom.atom_type import AtomType
from external_asset_ism_ismc_generation_tool.media_data_parser.atom_parser.trak_parser import TRAKParser
from external_asset_ism_ismc_generation_tool.media_data_parser.atom_parser.stts_parser import STTSParser
from external_asset_ism_ismc_generation_tool.media_data_parser.media_data_parser import MediaDataParser
from external_asset_ism_ismc_generation_tool.media_data_parser.media_track_info_extractor import MediaTrackInfoExtractor
from external_asset_ism_ismc_generation_tool.media_data_parser.model.media_track_info import MediaTrackInfo
from external_asset_ism_ismc_generation_tool.media_data_parser.model.track_type import TrackType
from external_asset_ism_ismc_generation_tool.mss_client_manifest.ismc_generator import IsmcGenerator


def _box(box_type: bytes, payload: bytes = b"") -> bytes:
    """Builds a standard ISO BMFF box with a 32-bit size field."""
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def _extended_box(box_type: bytes, payload: bytes = b"") -> bytes:
    """Builds an ISO BMFF box using the 64-bit extended size ('largesize') field."""
    size = 16 + len(payload)
    return struct.pack(">I", 1) + box_type + struct.pack(">Q", size) + payload


def _make_client(tmp_path, file_name: str, content: bytes) -> LocalFileServiceClient:
    (tmp_path / file_name).write_bytes(content)
    return LocalFileServiceClient({"local_directory": str(tmp_path)})


@pytest.mark.parametrize("version,entry_format", [(0, ">Ii hh"), (1, ">Qq hh")])
def test_edit_list_versions(version, entry_format):
    entries = struct.pack(entry_format, 1920, -1, 1, 0)
    media_duration = 2**33 if version == 1 else 3886120
    entries += struct.pack(entry_format, media_duration, 3600, 1, 0)
    payload = struct.pack(">B3xI", version, 2) + entries
    parsed = Box.parse(_box(b"edts", _box(b"elst", payload)))
    edits = parsed.children[0].entries
    assert edits[0].media_time == -1
    assert edits[0].segment_duration == 1920
    assert edits[1].media_time == 3600
    assert edits[1].segment_duration == media_duration


@pytest.mark.parametrize(
    "track_type,media_time,empty_duration,composition_offset,edit_duration,expected_start,expected_chunks",
    [
        (TrackType.AUDIO, 30720, 0, 0, 10240, 0, [3.2, 3.84, 2.56]),
        (TrackType.VIDEO, 1920, 1920, 1920, 10240, 1.92, [3.84, 3.84, 2.56]),
        (TrackType.AUDIO, 215040, 0, 0, 10240, 0, [3.2, 2.56]),
        (TrackType.AUDIO, 30720, 0, 0, 0, 0, [3.2, 3.84, 2.56]),
        (TrackType.AUDIO, 30720, 0, 0, 5000, 0, [3.2, 1.8]),
        (TrackType.VIDEO, 0, 0, 1920, 10240, 0.04, [3.84, 3.84, 2.52]),
    ]
)
def test_edit_list_maps_track_timeline(track_type, media_time, empty_duration,
                                      composition_offset, edit_duration, expected_start, expected_chunks):
    entries = []
    if empty_duration:
        entries.append(struct.pack(">Ii hh", empty_duration, -1, 1, 0))
    entries.append(struct.pack(">Ii hh", edit_duration, media_time, 1, 0))
    edts = Box.parse(_box(b"edts", _box(b"elst", struct.pack(">II", 0, len(entries)) + b"".join(entries))))
    parser = TRAKParser(SimpleNamespace(children=[edts]))
    parser.get_timescale = lambda: 48000
    extractor = MediaTrackInfoExtractor.__new__(MediaTrackInfoExtractor)
    extractor.trak_parser = parser
    extractor.movie_timescale = 1000
    extractor.timescale = 48000
    extractor.ctts_atom = SimpleNamespace(entries=[SimpleNamespace(sample_offset=composition_offset)])
    extractor.track_type = track_type
    extractor.blob_name = 'edits.mp4'
    track = MediaTrackInfo(track_type, '1000', 1, 3, 'AACL', [3.84, 3.84, 2.56], 'edits.mp4')
    extractor._MediaTrackInfoExtractor__extract_audio_track_info = lambda fragments: track
    extractor._MediaTrackInfoExtractor__extract_video_track_info = lambda fragments: track
    result = extractor.get_track_info({})
    assert result.start_time == Fraction(str(expected_start))
    assert result.chunk_datas == expected_chunks
    assert result.chunks == len(expected_chunks)
    manifest = ET.fromstring(IsmcGenerator.generate(12, [result], packager_timing_compatibility=False))
    assert manifest.find('StreamIndex/c').get('t') == str(round(expected_start * 10000000))


@pytest.mark.parametrize('source', ['stts', 'moof'])
@pytest.mark.parametrize('align', [False, True])
@pytest.mark.parametrize('timescale,chunk_ticks,media_time,edit_duration', [
    (48000, 102400, Fraction(0), Fraction(32, 5)),
    (44100, 89088, Fraction(89088, 44100), Fraction(0)),
])
def test_edit_list_exact_chunk_boundaries(timescale, chunk_ticks, media_time, edit_duration, source, align):
    if source == 'stts':
        stts = STTSParser({'entries': [SimpleNamespace(sample_count=4 * (chunk_ticks // 1024), sample_delta=1024)]})
        chunks = stts.get_chunk_durations_from_stts(
            TrackType.AUDIO, timescale, segment_duration_ticks=chunk_ticks)
    else:
        fragments = {}
        tfhd = SimpleNamespace(track_ID=1)
        trun = SimpleNamespace(sample_info=[SimpleNamespace(sample_duration=chunk_ticks, sample_size=1000)])
        for fragment_number in range(4):
            MediaDataParser._MediaDataParser__fill_moof_fragment(fragments, tfhd, trun, None, timescale)
        chunks = fragments[1][0]

    extractor = MediaTrackInfoExtractor.__new__(MediaTrackInfoExtractor)
    extractor.trak_parser = SimpleNamespace(
        get_edit_timing=lambda timescale: (Fraction(0), media_time, edit_duration))
    extractor.movie_timescale = 1000
    extractor.ctts_atom = None
    extractor.track_type = TrackType.AUDIO
    extractor.blob_name = 'exact-boundary.mp4'
    track = MediaTrackInfo(TrackType.AUDIO, '1000', 1, len(chunks), 'AACL', chunks, extractor.blob_name)
    extractor._MediaTrackInfoExtractor__extract_audio_track_info = lambda fragments: track

    result = extractor.get_track_info({})
    assert result.chunks == 3
    assert chunks == [Fraction(chunk_ticks, timescale)] * 4
    assert result.chunk_datas == [float(Fraction(chunk_ticks, timescale))] * 3
    manifest = ET.fromstring(IsmcGenerator.generate(10, [result], packager_timing_compatibility=align))
    stream = manifest.find('StreamIndex')
    assert stream.get('Chunks') == '3'
    assert all(int(chunk.get('d')) > 0 for chunk in stream.findall('c'))


@pytest.mark.parametrize('media_time,expected_start,expected_duration', [
    (Fraction(2, 25), Fraction(0), Fraction(2)),
    (Fraction(3, 25), Fraction(0), Fraction(49, 25)),
    (Fraction(1, 25), Fraction(1, 25), Fraction(2)),
])
def test_fragment_composition_offset_is_preserved_during_edit_trimming(
        media_time, expected_start, expected_duration):
    fragments = {}
    timings = {}
    tfhd = SimpleNamespace(track_ID=1)
    samples = [SimpleNamespace(sample_duration=80, sample_size=100,
                               sample_composition_time_offsets=80) for sample_number in range(25)]
    trun = SimpleNamespace(sample_info=samples)
    MediaDataParser._MediaDataParser__fill_moof_fragment(
        fragments, tfhd, trun, None, 1000, timings)
    extractor = MediaTrackInfoExtractor.__new__(MediaTrackInfoExtractor)
    extractor.trak_parser = SimpleNamespace(
        get_edit_timing=lambda timescale: (Fraction(0), media_time, Fraction(0)))
    extractor.movie_timescale = 1000
    extractor.ctts_atom = None
    extractor.track_type = TrackType.VIDEO
    extractor.blob_name = 'fragment-edits.mp4'
    track = MediaTrackInfo(TrackType.VIDEO, '1000', 1, 1, 'AVC1',
                           fragments[1][0], extractor.blob_name)
    extractor._MediaTrackInfoExtractor__extract_video_track_info = lambda fragments: track

    result = extractor.get_track_info(fragments, timings)

    assert result.start_time == expected_start
    assert result.chunk_datas == [expected_duration]
    assert result.chunks == 1


def _fragmented_video_moov(tracks=((1, 1000),), edits=None, movie_duration=0, movie_timescale=1000):
    track_boxes = []
    defaults = []
    for track_id, timescale in tracks:
        sample_entry = (bytes(6) + struct.pack('>H', 1) + bytes(16)
                        + struct.pack('>HHII', 16, 16, 0x480000, 0x480000)
                        + bytes(4) + struct.pack('>H', 1) + bytes(32) + struct.pack('>Hh', 24, -1))
        stsd = _box(b'stsd', struct.pack('>II', 0, 1) + _box(b'avc1', sample_entry))
        stbl = _box(b'stbl', stsd + _box(b'stts', bytes(8)) + _box(b'stsz', bytes(12)))
        mdhd = _box(b'mdhd', struct.pack('>IIIIIHH', 0, 0, 0, timescale, 0, 0x55C4, 0))
        hdlr = _box(b'hdlr', bytes(8) + b'vide' + bytes(12) + b'Video\0')
        mdia = _box(b'mdia', mdhd + hdlr + _box(b'minf', stbl))
        edts = b''
        if edits is not None:
            entries = b''.join(struct.pack('>Ii hh', duration, media_time, 1, 0)
                               for duration, media_time in edits)
            edts = _box(b'edts', _box(b'elst', struct.pack('>II', 0, len(edits)) + entries))
        tkhd = _box(b'tkhd', struct.pack('>IIIIII', 1, 0, 0, track_id, 0, 0) + bytes(60))
        track_boxes.append(_box(b'trak', tkhd + edts + mdia))
        defaults.append(_box(b'trex', struct.pack('>IIIIII', 0, track_id, 1, timescale, 100, 0)))
        mvhd = _box(b'mvhd', struct.pack('>IIIII', 0, 0, 0, movie_timescale, movie_duration)
            + bytes(76) + struct.pack('>I', len(tracks) + 1))
    return _box(b'moov', mvhd + b''.join(track_boxes) + _box(b'mvex', b''.join(reversed(defaults))))


def _timed_fragment(track_id=1, decode_time=0, runs=(((1000, 0),),), version=0, tfdt_version=1):
    tfhd = _box(b'tfhd', struct.pack('>II', 0, track_id))
    tfdt = b'' if decode_time is None else _box(
        b'tfdt', struct.pack('>I', tfdt_version << 24)
        + struct.pack('>Q' if tfdt_version == 1 else '>I', decode_time))
    truns = []
    for samples in runs:
        explicit_duration = samples[0][0] is not None
        flags = 0x800 | (0x100 if explicit_duration else 0)
        payload = struct.pack('>II', version << 24 | flags, len(samples))
        for duration, offset in samples:
            if explicit_duration:
                payload += struct.pack('>I', duration)
            payload += struct.pack('>i' if version == 1 else '>I', offset)
        truns.append(_box(b'trun', payload))
    return _box(b'moof', _box(b'traf', tfhd + tfdt + b''.join(truns)))


@pytest.mark.parametrize('extended', [False, True])
def test_fragment_reader_retains_random_access_index_and_file_offsets(tmp_path, extended):
    moov = _fragmented_video_moov()
    fragment = _timed_fragment()
    body = _box(b'mfro', bytes(8))
    mfra = (_extended_box if extended else _box)(b'mfra', body)
    client = _make_client(tmp_path, 'indexed.mp4', moov + fragment + mfra)

    data = LocalMediaDataParser.get_media_data(client, 'indexed.mp4')

    assert data['mfra'] == _box(b'mfra', body)
    assert data['moof_offsets'] == [len(moov)]
    assert data['moofs'] == [fragment]


@pytest.mark.parametrize('tfdt_version', [0, 1])
def test_binary_fragment_edit_does_not_trim_composition_delay_twice(tfdt_version):
    data = {'moov': _fragmented_video_moov(edits=[(0, 80)]), 'moofs': [
        _timed_fragment(runs=(tuple((80, 80) for sample_number in range(25)),),
                        tfdt_version=tfdt_version)]}
    result = MediaDataParser.parse_media_data('video.mp4', data)
    track = result.media_track_info_list[0]
    assert track.start_time == 0
    assert track.chunk_datas == [2.0]
    assert track.end_time == 2
    assert result.media_duration == 2


def _tfra(entries, track_id=1, version=1, width=1):
    lengths = (width - 1) * 21
    payload = struct.pack('>IIII', version << 24, track_id, lengths, len(entries))
    for time, offset, traf, run, sample in entries:
        payload += struct.pack('>QQ' if version else '>II', time, offset)
        payload += b''.join(value.to_bytes(width, 'big') for value in (traf, run, sample))
    return _box(b'tfra', payload)


def _indexed_data(entries=None, version=1, width=1):
    if entries is None:
        entries = [(0, 500, 1, 1, 1), (2000, 900, 1, 1, 1)]
    return {
        'moov': _fragmented_video_moov(edits=[(0, 85)]),
        'moofs': [_timed_fragment(decode_time=start, runs=(((2000, 40),),)) for start in (0, 2000)],
        'moof_offsets': [500, 900],
        'mfra': _box(b'mfra', _tfra(entries, version=version, width=width)),
    }


@pytest.mark.parametrize('version', [0, 1])
@pytest.mark.parametrize('width', [1, 2, 3, 4])
def test_tfra_metadata_preserves_index_keys_separately_from_edited_timing(version, width):
    result = MediaDataParser.parse_media_data('indexed.mp4', _indexed_data(version=version, width=width))
    track = result.media_track_info_list[0]

    assert track.fragment_index == [(0, 2000), (2000, 2000)]
    assert track.fragment_timescale == 1000
    assert track.start_time == 0
    assert track.chunk_datas == [Fraction(391, 200), Fraction(2)]
    assert track.end_time == Fraction(791, 200)
    assert result.duration_bounds == (0, 4)


@pytest.mark.parametrize('entries,message', [
    ([(0, 501, 1, 1, 1), (2000, 900, 1, 1, 1)], 'fragment reference'),
    ([(0, 500, 2, 1, 1), (2000, 900, 1, 1, 1)], 'fragment reference'),
    ([(0, 500, 1, 2, 1), (2000, 900, 1, 1, 1)], 'fragment reference'),
    ([(0, 500, 1, 1, 2), (2000, 900, 1, 1, 1)], 'fragment reference'),
    ([(0, 500, 1, 1, 1)], 'Incomplete tfra'),
    ([(0, 500, 1, 1, 1), (1999, 900, 1, 1, 1)], 'duration does not match'),
    ([(2000, 500, 1, 1, 1), (0, 900, 1, 1, 1)], 'duration does not match'),
])
def test_tfra_rejects_unsupported_fragment_references(entries, message):
    with pytest.raises(ValueError, match=message):
        MediaDataParser.parse_media_data('indexed.mp4', _indexed_data(entries))


@pytest.mark.parametrize('mfra,message', [
    (b'', 'Malformed mfra'),
    (_box(b'free'), 'Malformed mfra'),
    (struct.pack('>I4s', 9, b'mfra'), 'Malformed mfra'),
    (_box(b'mfra', b'\0'), 'Truncated mfra child header'),
    (_box(b'mfra', struct.pack('>I4s', 1, b'tfra')), 'Truncated extended mfra child header'),
    (_box(b'mfra', struct.pack('>I4sQ', 1, b'tfra', 8)), 'Invalid mfra child size'),
    (_box(b'mfra', _box(b'tfra', struct.pack('>IIII', 1, 1, 0, 0))), 'Unsupported tfra'),
    (_box(b'mfra', _box(b'tfra', struct.pack('>IIII', 0, 1, 64, 0))), 'Unsupported tfra'),
    (_box(b'mfra'), 'no tfra'),
    (_box(b'mfra', _tfra([])), 'Empty or truncated'),
    (_box(b'mfra', _tfra([(0, 500, 1, 1, 1)], version=2)), 'Unsupported tfra'),
    (_box(b'mfra', _tfra([(0, 500, 1, 1, 1)], track_id=9)), 'Unknown tfra track'),
    (_box(b'mfra', _box(b'tfra', bytes(12))), 'Truncated tfra header'),
    (_box(b'mfra', struct.pack('>I4s', 100, b'tfra')), 'Invalid mfra child size'),
    (_box(b'mfra', _tfra([(0, 500, 1, 1, 1)]) + _tfra([(0, 500, 1, 1, 1)])), 'Duplicate tfra'),
])
def test_tfra_rejects_malformed_or_ambiguous_indexes(mfra, message):
    data = _indexed_data()
    data['mfra'] = mfra
    with pytest.raises(ValueError, match=message):
        MediaDataParser.parse_media_data('indexed.mp4', data)


def test_tfra_supports_extended_children_and_skips_unrelated_boxes():
    data = _indexed_data()
    data['mfra'] = _box(b'mfra', _box(b'free') + _extended_box(b'tfra', data['mfra'][16:]))
    track = MediaDataParser.parse_media_data('indexed.mp4', data).media_track_info_list[0]
    assert track.fragment_index == [(0, 2000), (2000, 2000)]


@pytest.mark.parametrize('offsets', [None, [], [500], [500, 900, 1300]])
def test_tfra_requires_complete_fragment_offset_metadata(offsets):
    data = _indexed_data()
    if offsets is None:
        del data['moof_offsets']
    else:
        data['moof_offsets'] = offsets
    with pytest.raises(ValueError, match='Missing fragment file offsets.*indexed.mp4'):
        MediaDataParser.parse_media_data('indexed.mp4', data)


@pytest.mark.parametrize('include_fragments', [False, True])
def test_tfra_requires_an_index_for_every_av_track(include_fragments):
    data = _indexed_data()
    data['moov'] = _fragmented_video_moov(tracks=((1, 1000), (2, 1000)))
    if include_fragments:
        data['moofs'].append(_timed_fragment(track_id=2))
        data['moof_offsets'].append(1300)
    message = 'Incomplete tfra index.*track 2' if include_fragments else 'No fragments.*indexed.mp4.*track 2'
    with pytest.raises(ValueError, match=message):
        MediaDataParser.parse_media_data('indexed.mp4', data)


def test_tfra_rejects_decreasing_physical_offsets_even_when_references_match():
    data = _indexed_data([(0, 900, 1, 1, 1), (2000, 500, 1, 1, 1)])
    data['moof_offsets'] = [900, 500]
    with pytest.raises(ValueError, match='Non-increasing tfra index'):
        MediaDataParser.parse_media_data('indexed.mp4', data)


def test_tfra_tracks_match_interleaved_fragments_and_track_defaults():
    data = {
        'moov': _fragmented_video_moov(tracks=((7, 1000), (3, 48000))),
        'moofs': [
            _timed_fragment(7, 10000, runs=(((None, 80),), ((None, 80),))),
            _timed_fragment(3, 480000, runs=(((None, 0),),)),
            _timed_fragment(7, None, runs=(((None, 80),),)),
        ],
        'moof_offsets': [500, 900, 1300],
        'mfra': _box(b'mfra',
                     _tfra([(480000, 900, 1, 1, 1)], track_id=3)
                     + _tfra([(10080, 500, 1, 1, 1), (12080, 1300, 1, 1, 1)], track_id=7)),
    }
    result = MediaDataParser.parse_media_data('tracks.mp4', data)
    first, second = result.media_track_info_list

    assert first.fragment_index == [(10080, 2000), (12080, 1000)]
    assert second.fragment_index == [(480000, 48000)]
    root = ET.fromstring(IsmcGenerator.generate(result.media_duration, [first, second]))
    streams = root.findall('StreamIndex')
    assert [stream.get('TimeScale') for stream in streams] == ['1000', '48000']
    assert [stream.find('c').get('t') for stream in streams] == ['10080', '480000']


@pytest.mark.parametrize('align', [False, True])
@pytest.mark.parametrize('index_start', [0, 40, 2**54 + 1])
def test_indexed_manifest_uses_native_keys_without_edit_trimming_or_alignment(align, index_start):
    entries = [(index_start, 500, 1, 1, 1), (index_start + 2000, 900, 1, 1, 1)]
    data = _indexed_data(entries)
    data['moov'] = _fragmented_video_moov(tracks=((1, 44100),), edits=[(0, 85)])
    result = MediaDataParser.parse_media_data('indexed.mp4', data)
    track = result.media_track_info_list[0]
    original_chunks = list(track.chunk_datas)
    root = ET.fromstring(IsmcGenerator.generate(result.media_duration, [track], packager_timing_compatibility=align))
    stream = root.find('StreamIndex')

    assert root.get('TimeScale') == '10000000'
    assert stream.get('TimeScale') == '44100'
    assert stream.get('StreamStartTimestamp') == ('1' if align else None)
    assert stream.get('Chunks') == '2'
    assert [chunk.attrib for chunk in stream.findall('c')] == [
        {'t': str(index_start), 'd': '2000', 'r': '2'}]
    assert track.start_time == 0
    assert track.chunk_datas == original_chunks
    assert track.chunk_datas[0] < Fraction(2000, 44100)


@pytest.mark.parametrize('align', [False, True])
def test_indexed_hls_marker_uses_common_presentation_start_without_changing_addresses(align):
    audio = MediaTrackInfo(TrackType.AUDIO, '1000', 2, 1, 'AACL', [2], 'audio.mp4')
    audio.fragment_index = [(0, 96000)]
    audio.fragment_timescale = 48000
    video = MediaTrackInfo(TrackType.VIDEO, '2000', 1, 1, 'AVC1', [2], 'video.mp4',
                           start_time=Fraction(1, 25))
    video.fragment_index = [(0, 180000)]
    video.fragment_timescale = 90000

    root = ET.fromstring(IsmcGenerator.generate(2, [audio, video], packager_timing_compatibility=align))
    streams = root.findall('StreamIndex')

    assert [stream.get('StreamStartTimestamp') for stream in streams] == (
        ['1920', '3600'] if align else [None, None])
    assert [stream.find('c').attrib for stream in streams] == [
        {'t': '0', 'd': '96000', 'r': '1'}, {'t': '0', 'd': '180000', 'r': '1'}]
    assert audio.start_time == 0
    assert video.start_time == Fraction(1, 25)
    assert audio.chunk_datas == video.chunk_datas == [2]


@pytest.mark.parametrize('align', [False, True])
def test_indexed_renditions_group_by_source_index_not_different_edits(align):
    first = MediaDataParser.parse_media_data('first.mp4', _indexed_data()).media_track_info_list[0]
    data = _indexed_data()
    data['moov'] = _fragmented_video_moov(edits=[(0, 100)])
    second = MediaDataParser.parse_media_data('second.mp4', data).media_track_info_list[0]
    second.bit_rate = str(int(first.bit_rate) + 1)
    assert first.chunk_datas != second.chunk_datas

    root = ET.fromstring(IsmcGenerator.generate(4, [first, second], packager_timing_compatibility=align))

    assert len(root.findall('StreamIndex')) == 1
    assert root.find('StreamIndex').get('QualityLevels') == '2'


@pytest.mark.parametrize('offsets,version,expected_start,expected_end', [
    ([80, 80, 80], 0, Fraction(27, 25), Fraction(33, 25)),
    ([-80, -80, -80], 1, Fraction(23, 25), Fraction(29, 25)),
    ([160, 0, -160], 1, Fraction(1), Fraction(31, 25)),
])
def test_binary_fragments_use_absolute_presentation_times_and_reordering(
        offsets, version, expected_start, expected_end):
    data = {'moov': _fragmented_video_moov(), 'moofs': [
        _timed_fragment(decode_time=1000, runs=(tuple((80, offset) for offset in offsets),), version=version)]}
    result = MediaDataParser.parse_media_data('video.mp4', data)
    track = result.media_track_info_list[0]
    assert track.start_time == expected_start
    assert track.end_time == expected_end
    assert track.chunk_datas == [Fraction(6, 25)]
    manifest = ET.fromstring(IsmcGenerator.generate(result.media_duration, [track], packager_timing_compatibility=False))
    assert manifest.find('StreamIndex/c').get('t') == str(round(expected_start * 10000000))
    assert manifest.find('StreamIndex').get('StreamStartTimestamp') is None


def test_binary_fragment_tracks_use_their_own_defaults_and_combine_runs_once():
    data = {'moov': _fragmented_video_moov(tracks=((7, 1000), (3, 48000))), 'moofs': [
        _timed_fragment(7, 10000, runs=(((None, 80),), ((None, 80),))),
        _timed_fragment(3, 480000, runs=(((None, 0),),)),
        _timed_fragment(7, None, runs=(((None, 80),),)),
    ]}
    result = MediaDataParser.parse_media_data('tracks.mp4', data)
    video, other = result.media_track_info_list
    assert video.track_id == 7
    assert video.start_time == Fraction(252, 25)
    assert video.chunk_datas == [2.0, 1.0]
    assert other.track_id == 3
    assert other.start_time == 10
    assert other.chunk_datas == [1.0]
    assert result.media_duration == Fraction(77, 25)


@pytest.mark.parametrize('second_start', [900, 1100])
def test_binary_fragment_decode_gaps_and_overlaps_fail_explicitly(second_start):
    data = {'moov': _fragmented_video_moov(), 'moofs': [
        _timed_fragment(), _timed_fragment(decode_time=second_start)]}
    with pytest.raises(ValueError, match='decode gap or overlap.*track 1'):
        MediaDataParser.parse_media_data('gap.mp4', data)


def test_binary_fragment_presentation_discontinuities_fail_explicitly():
    data = {'moov': _fragmented_video_moov(), 'moofs': [
        _timed_fragment(), _timed_fragment(decode_time=1000, runs=(((1000, 80),),))]}
    with pytest.raises(ValueError, match='presentation gap or overlap.*track 1'):
        MediaDataParser.parse_media_data('gap.mp4', data)


@pytest.mark.parametrize('fragment,message', [
    (_box(b'moof', _box(b'traf')), 'missing or unknown track header'),
    (_timed_fragment(track_id=9), 'missing or unknown track header'),
    (_box(b'moof', _box(b'traf', _box(b'tfhd', struct.pack('>II', 0, 1)))), 'Empty fragment'),
    (_timed_fragment(runs=(((0, 0),),)), 'Nonpositive sample duration'),
    (_timed_fragment(runs=(((1000, -40),),), version=1), 'Negative fragment presentation start'),
    (_timed_fragment(runs=(((500, 0), (500, 40)),)), 'composition gap or overlap'),
    (_timed_fragment(runs=(((500, 0), (500, -40)),), version=1), 'composition gap or overlap'),
])
def test_binary_fragment_invalid_timing_fails_explicitly(fragment, message):
    data = {'moov': _fragmented_video_moov(), 'moofs': [fragment]}
    with pytest.raises(ValueError, match=message):
        MediaDataParser.parse_media_data('invalid.mp4', data)


def test_binary_fragments_require_a_positive_track_timescale():
    data = {'moov': _fragmented_video_moov(tracks=((1, 0),)), 'moofs': [_timed_fragment()]}
    with pytest.raises(ValueError, match='positive timescale.*track 1'):
        MediaDataParser.parse_media_data('invalid.mp4', data)


def test_binary_fragment_edit_that_selects_no_media_fails_explicitly():
    data = {'moov': _fragmented_video_moov(edits=[(0, 10000)]), 'moofs': [_timed_fragment()]}
    with pytest.raises(ValueError, match='Edit list selects no media.*invalid.mp4.*track 1'):
        MediaDataParser.parse_media_data('invalid.mp4', data)


@pytest.mark.parametrize('edits,expected_start,expected_duration', [
    ([(200, -1), (500, 300)], Fraction(1, 5), Fraction(1, 2)),
    ([(0, 80)], Fraction(0), Fraction(2)),
])
def test_binary_fragment_edits_clip_media_coordinates(edits, expected_start, expected_duration):
    data = {'moov': _fragmented_video_moov(edits=edits), 'moofs': [
        _timed_fragment(runs=(tuple((80, 80) for sample_number in range(25)),))]}
    result = MediaDataParser.parse_media_data('edits.mp4', data)
    track = result.media_track_info_list[0]
    assert track.start_time == expected_start
    assert track.chunk_datas == [expected_duration]
    assert track.end_time == expected_start + expected_duration


@pytest.mark.parametrize('multithreading', [False, True])
def test_fragment_duration_aggregation_preserves_relative_file_origins(multithreading):
    inputs = {
        'early.mp4': {'moov': _fragmented_video_moov(),
                      'moofs': [_timed_fragment(decode_time=10000)]},
        'late.mp4': {'moov': _fragmented_video_moov(edits=[(200, -1), (0, 80)]),
                     'moofs': [_timed_fragment(decode_time=10500, runs=(((1000, 80),),))]},
    }
    result = MediaDataParser.get_media_data(inputs, is_multithreading=multithreading)
    assert result.duration_bounds == (Fraction(10), Fraction(117, 10))
    assert result.media_duration == Fraction(17, 10)
    assert sorted(track.start_time for track in result.media_track_info_list) == [Fraction(10), Fraction(107, 10)]


def test_edit_list_rejects_a_media_start_beyond_the_track():
    extractor = MediaTrackInfoExtractor.__new__(MediaTrackInfoExtractor)
    extractor.trak_parser = SimpleNamespace(get_edit_timing=lambda timescale: (Fraction(0), Fraction(10), Fraction(1)))
    extractor.movie_timescale = 1000
    extractor.ctts_atom = None
    extractor.track_type = TrackType.AUDIO
    extractor.blob_name = 'empty-edit.mp4'
    track = MediaTrackInfo(TrackType.AUDIO, '1000', 2, 1, 'AACL', [2], 'empty-edit.mp4')
    extractor._MediaTrackInfoExtractor__extract_audio_track_info = lambda fragments: track
    with pytest.raises(ValueError, match='Edit list selects no media.*empty-edit.mp4.*track 2'):
        extractor.get_track_info({})


@pytest.mark.parametrize("version,offset", [(0, 3600), (1, -3600)])
def test_composition_offset_signedness(version, offset):
    entry_format = ">Ii" if version == 1 else ">II"
    payload = struct.pack(">B3xI", version, 1) + struct.pack(entry_format, 5, offset)
    assert Box.parse(_box(b"ctts", payload)).entries[0].sample_offset == offset


@pytest.mark.parametrize("align", [False, True])
def test_room02_manifest_timing_modes(align):
    audio = MediaTrackInfo(TrackType.AUDIO, '125374', 2, 1013, 'AACL',
                           [3.2] + [3.84] * 1011 + [2.56], 'room02.mp4')
    video = MediaTrackInfo(TrackType.VIDEO, '3998251', 1, 1013, 'AVC1',
                           [3.84] * 1012 + [0.04], 'room02.mp4', start_time=Fraction(48, 25))
    manifest = ET.fromstring(IsmcGenerator.generate(3888.04, [audio, video], packager_timing_compatibility=align))
    audio_stream, video_stream = manifest.findall('StreamIndex')
    assert manifest.get('Duration') == '38880400000'
    assert audio_stream.get('Chunks') == video_stream.get('Chunks') == '1013'
    assert [chunk.attrib for chunk in audio_stream.findall('c')] == [
        {'t': '19200000' if align else '0', 'd': '12800000' if align else '32000000', 'r': '1'},
        {'d': '38400000', 'r': '1011'},
        {'d': '25600000', 'r': '1'},
    ]
    assert [chunk.attrib for chunk in video_stream.findall('c')] == [
        {'t': '19200000', 'd': '38400000', 'r': '1012'},
        {'d': '400000', 'r': '1'},
    ]
    for stream in (audio_stream, video_stream):
        assert stream.get('StreamStartTimestamp') == ('19200000' if align else None)
    assert audio.start_time == 0
    assert audio.chunk_datas == [3.2] + [3.84] * 1011 + [2.56]


def test_alignment_discards_whole_chunks_and_preserves_remaining_boundaries():
    audio = MediaTrackInfo(TrackType.AUDIO, '1000', 2, 4, 'AACL', [2, 2, 2, 2], 'audio.mp4')
    video = MediaTrackInfo(TrackType.VIDEO, '2000', 1, 2, 'AVC1', [2, 2], 'video.mp4', start_time=5)
    text = MediaTrackInfo(TrackType.TEXT, '100', 3, 1, 'IMSC', [8], 'subs.cmft')
    manifest = ET.fromstring(IsmcGenerator.generate(9, [audio, video, text]))
    audio_stream, video_stream, text_stream = manifest.findall('StreamIndex')
    assert audio_stream.get('Chunks') == '2'
    assert [chunk.attrib for chunk in audio_stream.findall('c')] == [
        {'t': '50000000', 'd': '10000000', 'r': '1'},
        {'d': '20000000', 'r': '1'},
    ]
    assert video_stream.find('c').get('t') == '50000000'
    assert text_stream.find('c').get('t') == '0'
    assert text_stream.get('StreamStartTimestamp') is None
    standard = ET.fromstring(IsmcGenerator.generate(9, [audio, video, text], packager_timing_compatibility=False))
    assert standard.find('StreamIndex/c').attrib == {'t': '0', 'd': '20000000', 'r': '4'}


@pytest.mark.parametrize('source_start', [Fraction(0), Fraction(1, 8)])
@pytest.mark.parametrize('chunk_count,alignment_tick,expected_endpoints', [
    (4, 20266667, [40533333, 60800000, 81066667]),
    (4, 60800000, [81066667]),
    (4, 60800001, [81066667]),
    (4, 81066666, [81066667]),
    (4, 81066667, []),
    (4, 81066668, []),
    (3, 60800000, []),
    (3, 60800001, []),
])
def test_alignment_uses_rounded_cumulative_boundaries(source_start, chunk_count, alignment_tick, expected_endpoints):
    timescale = 10000000
    source_start_tick = round(source_start * timescale)
    aligned_start = source_start_tick + alignment_tick
    audio = MediaTrackInfo(TrackType.AUDIO, '1000', 2, chunk_count, 'AACL',
                           [97280 / 48000] * chunk_count, 'audio.mp4', start_time=source_start)
    video = MediaTrackInfo(TrackType.VIDEO, '2000', 1, 1, 'AVC1', [2], 'video.mp4',
                           start_time=Fraction(aligned_start, timescale))
    if not expected_endpoints:
        with pytest.raises(ValueError, match='No media remains.*audio.mp4'):
            IsmcGenerator.generate(12, [audio, video])
        return

    manifest = ET.fromstring(IsmcGenerator.generate(12, [audio, video]))
    stream = manifest.find('StreamIndex')
    assert stream.get('Chunks') == str(len(expected_endpoints))
    assert stream.find('c').get('t') == str(aligned_start)
    endpoints = []
    current_tick = aligned_start
    for chunk in stream.findall('c'):
        duration = int(chunk.get('d'))
        assert duration > 0
        for repeat_number in range(int(chunk.get('r'))):
            current_tick += duration
            endpoints.append(current_tick)
    assert endpoints == [source_start_tick + endpoint for endpoint in expected_endpoints]


def test_alignment_rejects_nonoverlapping_tracks():
    audio = MediaTrackInfo(TrackType.AUDIO, '1000', 2, 1, 'AACL', [2], 'audio.mp4')
    video = MediaTrackInfo(TrackType.VIDEO, '2000', 1, 1, 'AVC1', [2], 'video.mp4', start_time=3)
    with pytest.raises(ValueError, match='No media remains'):
        IsmcGenerator.generate(5, [audio, video])


@pytest.mark.parametrize('align', [False, True])
@pytest.mark.parametrize('track_id,language,chunks,merge_aligned', [
    (1, 'eng', [1, 2, 2], True),
    (2, 'eng', [1, 2, 2], False),
    (1, 'fra', [1, 2, 2], False),
    (1, 'eng', [2, 1, 2], False),
])
def test_renditions_group_by_effective_aligned_timeline(align, track_id, language, chunks, merge_aligned):
    early = MediaTrackInfo(TrackType.VIDEO, '1000', 1, 3, 'AVC1', [2, 2, 2], 'early.mp4', language='eng')
    late = MediaTrackInfo(TrackType.VIDEO, '2000', track_id, 3, 'AVC1', chunks, 'late.mp4',
                          language=language, start_time=1)
    original_chunks = [list(track.chunk_datas) for track in (early, late)]
    original_metadata = [vars(track).copy() for track in (early, late)]
    manifest = ET.fromstring(IsmcGenerator.generate(6, [early, late], packager_timing_compatibility=align))
    streams = manifest.findall('StreamIndex')
    if align and merge_aligned:
        assert len(streams) == 1
        assert streams[0].get('QualityLevels') == '2'
        assert [level.get('Bitrate') for level in streams[0].findall('QualityLevel')] == ['1000', '2000']
        assert [chunk.attrib for chunk in streams[0].findall('c')] == [
            {'t': '10000000', 'd': '10000000', 'r': '1'},
            {'d': '20000000', 'r': '2'},
        ]
    else:
        assert len(streams) == 2
        assert [stream.get('QualityLevels') for stream in streams] == ['1', '1']
    assert [vars(track) for track in (early, late)] == original_metadata
    assert [track.chunk_datas for track in (early, late)] == original_chunks


def test_tracks_with_different_presentation_starts_are_not_grouped():
    early = MediaTrackInfo(TrackType.VIDEO, '1000', 1, 1, 'AVC1', [2], 'early.mp4')
    late = MediaTrackInfo(TrackType.VIDEO, '2000', 1, 1, 'AVC1', [2], 'late.mp4', start_time=1)
    manifest = ET.fromstring(IsmcGenerator.generate(3, [early, late], packager_timing_compatibility=False))
    assert len(manifest.findall('StreamIndex')) == 2


@pytest.mark.parametrize("entries", [
    [(1000, 0, 0, 0)],
    [(1000, 0, 1, 1)],
    [(1000, -2, 1, 0)],
    [(1000, 0, 1, 0), (1000, 48000, 1, 0)],
    [(1000, 0, 1, 0), (1000, -1, 1, 0)],
    [(1000, -1, 1, 0)],
])
def test_unsupported_edit_lists_fail_explicitly(entries):
    payload = struct.pack(">II", 0, len(entries))
    payload += b''.join(struct.pack(">Ii hh", *entry) for entry in entries)
    edts = Box.parse(_box(b"edts", _box(b"elst", payload)))
    parser = TRAKParser(SimpleNamespace(children=[edts]))
    parser.get_timescale = lambda: 48000
    with pytest.raises(ValueError, match='edit'):
        parser.get_edit_timing(1000)


@pytest.mark.parametrize("payload", [
    b'',
    _box(b'elst', struct.pack('>II', 0, 1)),
    _box(b'elst', struct.pack('>II', 2 << 24, 0)),
])
def test_missing_or_malformed_edit_list_is_not_silently_ignored(payload):
    edts = Box.parse(_box(b'edts', payload))
    parser = TRAKParser(SimpleNamespace(children=[edts]))
    with pytest.raises(ValueError, match='missing or malformed edit list'):
        parser.get_edit_timing(1000)


def test_absent_or_empty_edit_list_has_no_timing_effect():
    assert TRAKParser(SimpleNamespace(children=[])).get_edit_timing(1000) is None
    edts = Box.parse(_box(b'edts', _box(b'elst', struct.pack('>II', 0, 0))))
    assert TRAKParser(SimpleNamespace(children=[edts])).get_edit_timing(1000) is None


@pytest.mark.parametrize('movie_timescale,media_timescale', [(0, 48000), (-1, 48000), (1000, 0), (1000, -1)])
def test_edit_lists_require_positive_timescales(movie_timescale, media_timescale):
    payload = struct.pack('>II', 0, 1) + struct.pack('>Ii hh', 1000, 0, 1, 0)
    edts = Box.parse(_box(b'edts', _box(b'elst', payload)))
    parser = TRAKParser(SimpleNamespace(children=[edts]))
    parser.get_timescale = lambda: media_timescale
    with pytest.raises(ValueError, match='positive movie and media timescales'):
        parser.get_edit_timing(movie_timescale)


def test_multiple_edit_list_boxes_are_rejected():
    elst = _box(b'elst', struct.pack('>II', 0, 0))
    parser = TRAKParser(SimpleNamespace(children=[Box.parse(_box(b'edts', elst + elst))]))
    with pytest.raises(ValueError, match='exactly one edit list'):
        parser.get_edit_timing(1000)


class TestExtendedBoxSize:
    """Verifies that boxes using the 64-bit extended size are parsed correctly."""

    def test_get_media_data_finds_moov_after_extended_size_mdat(self, tmp_path):
        # A non-fragmented file where 'mdat' precedes 'moov' and uses the
        # extended (largesize) box header, as seen with some encoders/muxers.
        ftyp = _box(b"ftyp", b"isom")
        mdat = _extended_box(b"mdat", b"D" * 64)
        moov = _box(b"moov", b"\x00" * 16)  # no mvex -> non-fragmented path

        file_name = "extended_size.mp4"
        client = _make_client(tmp_path, file_name, ftyp + mdat + moov)

        media_data = LocalMediaDataParser.get_media_data(client, file_name)

        assert media_data[AtomType.MOOV_ATOM_TYPE.value] == moov
        assert media_data["moofs"] == []

    def test_get_moov_data_reads_only_the_moov_box(self, tmp_path):
        ftyp = _box(b"ftyp", b"isom")
        moov = _box(b"moov", b"\x00" * 16)
        client = _make_client(tmp_path, "moov_only.mp4", ftyp + moov)

        media_data = LocalMediaDataParser.get_moov_data(client, "moov_only.mp4")

        assert media_data == {"moov": moov, "moofs": []}

    def test_find_and_process_moof_atoms_skips_extended_size_mdat(self, tmp_path):
        # A fragmented file where an extended-size 'mdat' sits between two
        # 'moof' boxes; the in-memory moof scan must resolve 'largesize' to
        # locate the second 'moof' correctly instead of misreading it as size 1.
        ftyp = _box(b"ftyp", b"isom")
        moov = _box(b"moov", b"mvex" + b"\x00" * 8)
        moof1 = _box(b"moof", b"F" * 16)
        mdat = _extended_box(b"mdat", b"D" * 64)
        moof2 = _box(b"moof", b"G" * 16)
        mfra = _box(b"mfra")

        file_name = "fragmented_extended_size.mp4"
        client = _make_client(tmp_path, file_name, ftyp + moov + moof1 + mdat + moof2 + mfra)

        media_data = LocalMediaDataParser.get_media_data(client, file_name)

        assert media_data[AtomType.MOOV_ATOM_TYPE.value] == moov
        assert media_data["moofs"] == [moof1, moof2]

    def test_get_media_data_raises_for_invalid_atom_size(self, tmp_path):
        # A box declaring size 0 ("extends to end of file") is not supported
        # and must be reported clearly instead of corrupting the scan.
        ftyp = _box(b"ftyp", b"isom")
        invalid_box = struct.pack(">I", 0) + b"free"
        moov = _box(b"moov")

        file_name = "invalid_size.mp4"
        client = _make_client(tmp_path, file_name, ftyp + invalid_box + moov)

        with pytest.raises(Exception, match="Invalid atom size"):
            LocalMediaDataParser.get_media_data(client, file_name)

    def test_get_media_data_raises_for_truncated_extended_size(self, tmp_path):
        # The file ends before the full 8-byte 'largesize' field is available;
        # a short read must not be silently decoded as a valid (smaller) size.
        ftyp = _box(b"ftyp", b"isom")
        truncated_mdat = struct.pack(">I", 1) + b"mdat" + b"\x00\x00\x00"  # only 3 of 8 largesize bytes

        file_name = "truncated_extended_size.mp4"
        client = _make_client(tmp_path, file_name, ftyp + truncated_mdat)

        with pytest.raises(Exception, match="Truncated extended size field"):
            LocalMediaDataParser.get_media_data(client, file_name)

    def test_find_and_process_moof_atoms_raises_for_truncated_extended_size(self, tmp_path):
        # Same truncated 'largesize' scenario, but hit via the in-memory moof scan.
        ftyp = _box(b"ftyp", b"isom")
        moov = _box(b"moov", b"mvex" + b"\x00" * 8)
        moof1 = _box(b"moof", b"F" * 16)
        truncated_mdat = struct.pack(">I", 1) + b"mdat" + b"\x00\x00\x00"  # only 3 of 8 largesize bytes

        file_name = "fragmented_truncated_extended_size.mp4"
        client = _make_client(tmp_path, file_name, ftyp + moov + moof1 + truncated_mdat)

        with pytest.raises(Exception, match="Truncated extended size field"):
            LocalMediaDataParser.get_media_data(client, file_name)

    def test_get_media_data_normalizes_extended_size_moov_header(self, tmp_path):
        # 'moov' itself (not just 'mdat') may use the extended size header;
        # downstream pymp4-based parsing only understands a standard 32-bit
        # size, so the returned bytes must always use the standard header.
        ftyp = _box(b"ftyp", b"isom")
        payload = b"\x00" * 32
        moov = _extended_box(b"moov", payload)

        file_name = "extended_moov.mp4"
        client = _make_client(tmp_path, file_name, ftyp + moov)

        media_data = LocalMediaDataParser.get_media_data(client, file_name)

        assert media_data[AtomType.MOOV_ATOM_TYPE.value] == _box(b"moov", payload)

    def test_find_and_process_moof_atoms_normalizes_extended_size_moof_header(self, tmp_path):
        # Same normalization requirement for an individual 'moof' fragment.
        ftyp = _box(b"ftyp", b"isom")
        moov = _box(b"moov", b"mvex" + b"\x00" * 8)
        moof1_payload = b"F" * 16
        moof1 = _extended_box(b"moof", moof1_payload)
        mfra = _box(b"mfra")

        file_name = "fragmented_extended_moof.mp4"
        client = _make_client(tmp_path, file_name, ftyp + moov + moof1 + mfra)

        media_data = LocalMediaDataParser.get_media_data(client, file_name)

        assert media_data["moofs"] == [_box(b"moof", moof1_payload)]

    def test_get_media_data_raises_for_truncated_atom_body(self, tmp_path):
        # 'moov' declares a size larger than the bytes actually available on
        # disk; a short read at EOF must not be silently accepted as a
        # smaller, valid box.
        ftyp = _box(b"ftyp", b"isom")
        oversized_moov_header = struct.pack(">I", 8 + 32) + b"moov"  # declares 32-byte payload
        actual_payload = b"\x00" * 10  # far fewer bytes actually present

        file_name = "truncated_moov_body.mp4"
        client = _make_client(tmp_path, file_name, ftyp + oversized_moov_header + actual_payload)

        with pytest.raises(Exception, match="Truncated atom"):
            LocalMediaDataParser.get_media_data(client, file_name)

    def test_find_and_process_moof_atoms_raises_for_atom_size_exceeding_buffer(self, tmp_path):
        # A box within the in-memory moof/fragment scan declares a size that
        # overruns the available buffer; slicing must not silently truncate
        # it into a corrupt-but-accepted box.
        ftyp = _box(b"ftyp", b"isom")
        moov = _box(b"moov", b"mvex" + b"\x00" * 8)
        moof1 = _box(b"moof", b"F" * 16)
        corrupt_header = struct.pack(">I", 1000) + b"free"  # declares far more than remains
        trailing_bytes = b"\x00" * 8

        file_name = "moof_declares_oversized_atom.mp4"
        client = _make_client(tmp_path, file_name, ftyp + moov + moof1 + corrupt_header + trailing_bytes)

        with pytest.raises(Exception, match="exceeds the available data"):
            LocalMediaDataParser.get_media_data(client, file_name)

    def test_build_standard_box_raises_for_body_exceeding_32bit_size(self):
        # A box whose body alone would push the normalized size past the
        # 32-bit header limit must be rejected explicitly instead of wrapping
        # around or producing a corrupt header. A fake body avoids allocating
        # multiple gigabytes just to exercise this boundary check.
        class _FakeHugeBody:
            def __len__(self):
                return 0xFFFFFFFF

        with pytest.raises(ValueError, match="is too large"):
            MediaFileDataReader.build_standard_box("moov", _FakeHugeBody())

    def test_scan_fragment_boxes_skips_huge_mdat_without_downloading_its_body(self):
        # A multi-gigabyte 'mdat' between two 'moof' boxes must be skipped via
        # offset arithmetic alone; it must never be read into memory.
        # A sparse fake file (only headers/moof bodies materialized) combined
        # with a hard cap on any single read length proves this holds even
        # for a file far too large to ever buffer in memory.
        class _SparseFileClient:
            def __init__(self, chunks, size, max_allowed_length):
                self._chunks = chunks
                self._size = size
                self._max_allowed_length = max_allowed_length

            def get_file_size(self, file_name):
                return self._size

            def download_part_of_file(self, file_name, offset=None, length=None):
                offset = offset or 0
                if length is not None and length > self._max_allowed_length:
                    raise AssertionError(
                        f"Unexpectedly requested {length} bytes at offset {offset}; "
                        f"large 'mdat' bodies must never be read"
                    )
                for chunk_offset, chunk in self._chunks.items():
                    if chunk_offset <= offset < chunk_offset + len(chunk):
                        rel = offset - chunk_offset
                        return chunk[rel:rel + length] if length is not None else chunk[rel:]
                raise AssertionError(f"Unexpected read at offset {offset} (no chunk registered there)")

        ftyp = _box(b"ftyp", b"isom")
        moov = _box(b"moov", b"mvex" + b"\x00" * 8)
        moof1 = _box(b"moof", b"F" * 16)
        offset_after_moof1 = len(ftyp) + len(moov) + len(moof1)

        huge_payload_len = 5 * 1024 ** 3  # 5 GiB, declared only -- never materialized
        mdat_total_size = 16 + huge_payload_len  # extended header + payload
        huge_mdat_header = struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", mdat_total_size)
        offset_after_mdat = offset_after_moof1 + mdat_total_size

        moof2 = _box(b"moof", b"G" * 16)
        mfra = _box(b"mfra")

        chunks = {
            0: ftyp + moov + moof1 + huge_mdat_header,
            offset_after_mdat: moof2 + mfra,
        }
        total_size = offset_after_mdat + len(moof2) + len(mfra)

        client = _SparseFileClient(chunks, total_size, max_allowed_length=64)

        media_data = LocalMediaDataParser.get_media_data(client, "huge_fragmented.mp4")

        assert media_data["moofs"] == [moof1, moof2]
