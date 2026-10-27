import math
from fractions import Fraction
from types import SimpleNamespace
from external_asset_ism_ismc_generation_tool.media_data_parser.atom_parser.fragment_index_parser import FragmentIndexParser
from typing import Any, Tuple, Dict, List, Union
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from tools.pymp4.src.pymp4.parser import Box

from external_asset_ism_ismc_generation_tool.common.logger.i_logger import ILogger
from external_asset_ism_ismc_generation_tool.common.logger.logger import Logger
from external_asset_ism_ismc_generation_tool.common.common import Common
from external_asset_ism_ismc_generation_tool.media_data_parser.media_box_extractor.media_box_extractor import MediaBoxExtractor
from external_asset_ism_ismc_generation_tool.media_data_parser.media_track_info_extractor import MediaTrackInfoExtractor
from external_asset_ism_ismc_generation_tool.media_data_parser.model.track_type import TrackType
from external_asset_ism_ismc_generation_tool.media_data_parser.model.media_track_info import MediaTrackInfo
from external_asset_ism_ismc_generation_tool.media_data_parser.model.media_data import MediaData
from external_asset_ism_ismc_generation_tool.media_data_parser.model.media_format import MediaFormat
from external_asset_ism_ismc_generation_tool.media_data_parser.atom_parser.nal_unit_type_parser import NalUnitTypeParser


class MediaDataParser:
    _MEDIA_HEADER_LENGTH = 8  # 8 bytes
    _MOOFS = 'moofs'
    __logger: ILogger = Logger("MediaDataParser")

    @classmethod
    def redefine_logger(cls, logger: ILogger):
        cls.__logger = logger

    @staticmethod
    def get_media_data(media_datas: Dict[str, dict], media_index_datas: Dict[str, dict] = None, is_multithreading: bool = False) -> MediaData:
        executor = None
        try:
            if is_multithreading:
                threads_num = cpu_count()
                executor = ProcessPoolExecutor(max_workers=threads_num)
            media_data: MediaData = MediaDataParser.__aggregate_media_data(media_datas, media_index_datas, executor)
            MediaDataParser.__update_media_track_info_list(media_data)

        finally:
            if executor:
                executor.shutdown()

        return media_data

    @staticmethod
    def parse_media_data(blob_name: str, media_data: Dict[str, Union[bytes, List[bytes]]]) -> MediaData:
        moof_fragments = {}
        moof_timings = {}
        media_track_info_list = []
        media_duration = 0

        parsed_moov_box = MediaBoxExtractor.extract_media_boxes(media_data["moov"])
        if not parsed_moov_box:
            MediaDataParser.__logger.error(f'Cannot parse moov box: {media_data["moov"]} for {blob_name}')
            raise ValueError("Cannot parse moov box")

        moov_atom = MediaBoxExtractor.get_mp4_box(parsed_moov_box, 'moov')
        if moov_atom:
            mvhd_atom = MediaBoxExtractor.get_mp4_sub_box(moov_atom, 'mvhd')
            media_duration = (Fraction(mvhd_atom['duration'], mvhd_atom['timescale'])
                              if mvhd_atom['timescale'] > 0 else 0)
            trak_atoms = MediaBoxExtractor.get_all_mp4_sub_boxes(moov_atom, 'trak')
            mvex_atom = MediaBoxExtractor.get_mp4_sub_box(moov_atom, 'mvex')
            mehd_atom = MediaBoxExtractor.get_mp4_sub_box(mvex_atom, 'mehd')
            if mehd_atom and mvhd_atom['timescale'] > 0:
                MediaDataParser.__logger.info(f'Moof boxes are detected in {blob_name}')
                media_duration = Fraction(mehd_atom["fragment_duration"], mvhd_atom['timescale'])
            trex_atoms = {atom.track_ID: atom for atom in
                          MediaBoxExtractor.get_all_mp4_sub_boxes(mvex_atom, 'trex')}
            extractors = [MediaTrackInfoExtractor(trak, mvhd_atom['duration'], mvhd_atom['timescale'],
                                                 blob_name, mvex_atom) for trak in trak_atoms]
            timescales = {extractor.track_id: extractor.timescale for extractor in extractors}
            fragment_indexes = (FragmentIndexParser.parse(media_data['mfra'])
                                if blob_name.lower().endswith('.mp4') and 'mfra' in media_data else None)
            if fragment_indexes is not None:
                if set(fragment_indexes) - set(timescales):
                    raise ValueError(f'Unknown tfra track in {blob_name}')
                if len(media_data.get('moof_offsets', [])) != len(media_data.get(MediaDataParser._MOOFS, [])):
                    raise ValueError(f'Missing fragment file offsets for {blob_name}')
            MediaDataParser.__fill_moof_fragments_from_boxes(
                media_data.get(MediaDataParser._MOOFS), moof_fragments, trex_atoms, timescales,
                moof_timings if blob_name.lower().endswith('.mp4') else None, media_data.get('moof_offsets'))

            # First pass: determine IDR-aligned segment duration for audio/subtitle alignment
            _SEGMENT_DURATION = 2  # seconds - same as in stts_parser
            video_segment_duration_ticks = None
            video_timescale = None
            idr_filtered_key_frames = None
            for trak_atom in trak_atoms:
                extractor = MediaTrackInfoExtractor(trak_atom, mvhd_atom['duration'], mvhd_atom['timescale'], blob_name, mvex_atom)
                if extractor.track_type == TrackType.VIDEO:
                    key_frames = extractor.stss_parser.get_key_frames_numbers_from_stss()

                    # Filter sync samples to true IDR/IRAP using NAL unit type parsing
                    sync_sample_headers = media_data.get('sync_sample_headers')
                    if key_frames and sync_sample_headers:
                        codec_type = extractor.stsd_parser.get_track_format()
                        nalu_length_size = extractor.stsd_parser.get_nalu_length_size()
                        if nalu_length_size:
                            idr_filtered_key_frames = NalUnitTypeParser.filter_idr_samples(
                                key_frames, sync_sample_headers, codec_type, nalu_length_size
                            )
                            MediaDataParser.__logger.info(
                                f'IDR filtering: {len(key_frames)} sync samples -> {len(idr_filtered_key_frames)} IDR samples for {blob_name}'
                            )

                    frames_for_period = idr_filtered_key_frames if idr_filtered_key_frames else key_frames
                    idr_period_ticks = extractor.stts_parser.get_idr_period_ticks(frames_for_period)
                    if idr_period_ticks is not None:
                        video_timescale = extractor.timescale
                        num_idr = math.ceil((_SEGMENT_DURATION * video_timescale) / idr_period_ticks)
                        video_segment_duration_ticks = num_idr * idr_period_ticks

                        # If IDR filtering was applied, generate complete IDR sample list
                        # from the detected period for use in chunk splitting.
                        # Intersect with the full STSS list to guard against ±1 sample
                        # tolerance drift producing positions that aren't actual sync samples.
                        if idr_filtered_key_frames and len(idr_filtered_key_frames) >= 2:
                            first_idr = int(idr_filtered_key_frames[0])
                            idr_interval = int(idr_filtered_key_frames[1]) - first_idr
                            total_samples = extractor.stts_parser.get_sample_count()
                            generated_positions = [
                                str(first_idr + i * idr_interval)
                                for i in range((total_samples - first_idr) // idr_interval + 1)
                            ]
                            # Only keep positions that are actual sync samples (present in STSS)
                            stss_set = set(key_frames)
                            idr_filtered_key_frames = [s for s in generated_positions if s in stss_set]
                            MediaDataParser.__logger.info(
                                f'Generated {len(idr_filtered_key_frames)} IDR sample numbers from period {idr_interval} '
                                f'(validated against {len(stss_set)} STSS entries) for {blob_name}'
                            )
                        else:
                            # Not enough IDR samples to derive a period-based list;
                            # discard partial subset to avoid degraded chunk splitting
                            idr_filtered_key_frames = None
                    else:
                        # No stable period detected; discard the IDR subset so chunk splitting
                        # falls back to the standard STSS-based or time-based segmentation
                        idr_filtered_key_frames = None
                    break

            for trak_atom in trak_atoms:
                media_track_info_creator = MediaTrackInfoExtractor(trak_atom, mvhd_atom['duration'], mvhd_atom['timescale'], blob_name, mvex_atom, video_segment_duration_ticks, video_timescale, idr_filtered_key_frames)
                track_info = media_track_info_creator.get_track_info(moof_fragments, moof_timings)
                if fragment_indexes is not None and track_info.track_type in (TrackType.AUDIO, TrackType.VIDEO):
                    MediaDataParser.__set_fragment_index(
                        track_info, fragment_indexes, moof_fragments, moof_timings, timescales[track_info.track_id])
                media_track_info_list.append(track_info)
        else:
            MediaDataParser.__logger.error(f'Cannot get tracks info: There is no `moov` atom in mp4 data for {blob_name}: {moov_atom}')
            raise ValueError("There is no 'moov' atom in mp4 data")
        duration_bounds = None
        timed_tracks = [track for track in media_track_info_list if track.end_time is not None]
        if media_duration <= 0 and timed_tracks:
            origin = min(Fraction(track.fragment_index[0][0], track.fragment_timescale)
                         if track.fragment_index is not None else min(track.decode_start_time, track.start_time)
                         for track in timed_tracks)
            endpoint = max(Fraction(sum(track.fragment_index[-1]), track.fragment_timescale)
                           if track.fragment_index is not None else track.end_time for track in timed_tracks)
            duration_bounds = origin, endpoint
            media_duration = endpoint - origin
        return MediaData(media_duration, media_track_info_list, duration_bounds)

    @staticmethod
    def __process_media_tasks_and_update_media_data(media_datas: Dict[str, dict], executor: ProcessPoolExecutor, media_data: MediaData):
        task_mapping = MediaDataParser.__map_media_tasks(media_datas, executor)

        for task in Common.get_completed_tasks(task_mapping, executor):
            blob_name = task_mapping[task] if executor else task
            try:
                task_media_data : MediaData = task.result() if executor else task_mapping[task]
                if task_media_data.media_duration > media_data.media_duration:
                    media_data.media_duration = task_media_data.media_duration
                if task_media_data.duration_bounds is not None:
                    origin, endpoint = task_media_data.duration_bounds
                    if media_data.duration_bounds is not None:
                        origin = min(origin, media_data.duration_bounds[0])
                        endpoint = max(endpoint, media_data.duration_bounds[1])
                    media_data.duration_bounds = origin, endpoint
                    media_data.media_duration = max(media_data.media_duration, endpoint - origin)
                if not MediaFormat.is_mpi_format(blob_name):
                    media_data.media_track_info_list += task_media_data.media_track_info_list
                else:
                    media_data.media_track_info_list = MediaDataParser.__update_media_track_info([media_data.media_track_info_list, task_media_data.media_track_info_list])

            except Exception as e:
                MediaDataParser.__logger.error(f"Error processing blob {blob_name}: {e}")
                raise ValueError(f"Error processing blob {blob_name}: {e}") from e

        media_data.media_track_info_list.sort(key=lambda track: (track.track_id, int(track.bit_rate)))

    @staticmethod
    def __aggregate_media_data(media_datas: Dict[str, dict], media_index_datas: Dict[str, dict], executor: ProcessPoolExecutor) -> MediaData:
        media_data = MediaData(0, [])

        MediaDataParser.__process_media_tasks_and_update_media_data(media_datas, executor, media_data)
        if media_index_datas:
            MediaDataParser.__process_media_tasks_and_update_media_data(media_index_datas, executor, media_data)

        return media_data

    @staticmethod
    def __map_media_tasks(media_datas: Dict[str, dict], executor: ProcessPoolExecutor) -> any:
        if executor:
            return {executor.submit(MediaDataParser.parse_media_data, blob_name, media_data): blob_name for blob_name, media_data in media_datas.items()}
        else:
            return {blob_name: MediaDataParser.parse_media_data(blob_name, media_data) for blob_name, media_data in media_datas.items()}

    @staticmethod
    def __update_media_track_info(track_info_lists: List[List[MediaTrackInfo]]) -> List[MediaTrackInfo]:
        media_track_info_list = track_info_lists[0]
        track_info_list = track_info_lists[1]
        for track in track_info_list:
            for media_track in media_track_info_list:
                if MediaDataParser.__should_change_track_info(track=media_track, track_index=track):
                    MediaDataParser.__change_track_info(track=media_track, track_index=track)
        return media_track_info_list

    @staticmethod
    def __get_moof_fragment_sample_total(trun_atom: Any, tfhd_atom: Any, trex_atom: Any,
                                         field_name: str) -> int:
        return sum(MediaDataParser.__get_moof_sample_value(sample, tfhd_atom, trex_atom, field_name)
                   for sample in trun_atom.sample_info)

    @staticmethod
    def __get_moof_sample_value(sample: Any, tfhd_atom: Any, trex_atom: Any, field_name: str) -> int:
        track_id = tfhd_atom.track_ID
        value = getattr(sample, field_name)
        if value is None:
            default_field_name = f'default_{field_name}'
            tfhd_default = getattr(tfhd_atom, default_field_name, None)
            trex_default = getattr(trex_atom, default_field_name, None) if trex_atom else None
            if tfhd_default is not None:
                value = tfhd_default
            elif trex_atom is not None and trex_atom.track_ID == track_id and trex_default not in (None, 0):
                value = trex_default
        if value is None:
            raise ValueError(
                f"Cannot calculate moof fragment {field_name}: sample value and defaults are missing for track {track_id}"
            )
        return value

    @staticmethod
    def __get_moof_fragment_duration(trun_atom: Any, tfhd_atom: Any, trex_atom: Any) -> int:
        return MediaDataParser.__get_moof_fragment_sample_total(
            trun_atom, tfhd_atom, trex_atom, 'sample_duration'
        )

    @staticmethod
    def __get_moof_fragment_size(trun_atom: Any, tfhd_atom: Any, trex_atom: Any) -> int:
        return MediaDataParser.__get_moof_fragment_sample_total(
            trun_atom, tfhd_atom, trex_atom, 'sample_size'
        )

    @staticmethod
    def __fill_moof_fragment(moof_fragments: Dict[int, List], tfhd_atom: Box, trun_atom: Box,
                             trex_atom: Box, timescale: int, moof_timings: dict = None,
                             decode_time: int = 0) -> None:
        track_id = tfhd_atom.track_ID
        fragment = moof_fragments.setdefault(track_id, [[], []])
        duration = MediaDataParser.__get_moof_fragment_duration(trun_atom, tfhd_atom, trex_atom)
        fragment[0].append(Fraction(duration, timescale))
        size = MediaDataParser.__get_moof_fragment_size(trun_atom, tfhd_atom, trex_atom)
        fragment[1].append(size)
        if moof_timings is not None:
            intervals = []
            for sample in trun_atom.sample_info:
                sample_duration = MediaDataParser.__get_moof_sample_value(
                    sample, tfhd_atom, trex_atom, 'sample_duration')
                offset = getattr(sample, 'sample_composition_time_offsets', None) or 0
                intervals.append((decode_time + offset, decode_time + offset + sample_duration))
                decode_time += sample_duration
            if not intervals:
                raise ValueError(f'Empty fragment for track {track_id}')
            ordered = sorted(intervals)
            for sample_start, sample_end in ordered:
                if sample_end <= sample_start:
                    raise ValueError(f'Nonpositive sample duration for track {track_id}')
            if any(previous[1] != current[0] for previous, current in zip(ordered, ordered[1:])):
                raise ValueError(f'Unsupported composition gap or overlap inside fragment for track {track_id}')
            moof_timings.setdefault(track_id, []).append((
                Fraction(min(start for start, end in intervals), timescale),
                Fraction(max(end for start, end in intervals), timescale)))

    @staticmethod
    def __should_change_track_info(track: MediaTrackInfo, track_index: MediaTrackInfo) -> bool:
        return track.track_id == track_index.track_id and track.codec_private_data == track_index.codec_private_data and track.bit_rate == track_index.bit_rate and\
            ((track.track_type == TrackType.VIDEO and track.width == track_index.width and track.height == track_index.height) or\
            (track.track_type == TrackType.AUDIO and track.sampling_rate == track_index.sampling_rate and track.channels == track_index.channels and track.language == track_index.language))

    @staticmethod
    def __change_track_info(track: MediaTrackInfo, track_index: MediaTrackInfo) -> None:
        track.index_blob_name = track_index.blob_name
        track.chunks = track_index.chunks
        track.chunk_datas = track_index.chunk_datas
        track.start_time = track_index.start_time
        track.end_time = track_index.end_time
        track.decode_start_time = track_index.decode_start_time
        track.fragment_index = track_index.fragment_index
        track.fragment_timescale = track_index.fragment_timescale
        track.bit_rate = track_index.bit_rate
        MediaDataParser.__logger.info(f'Changed chunks, bitrate, added index_blob_name {track.index_blob_name} for {track.blob_name} with track_id {track.track_id}')

    @staticmethod
    def __fill_moof_fragments_from_boxes(moof_boxes: List[bytes], moof_fragments: Dict[int, List],
                                       trex_atoms: dict, timescales: dict, moof_timings: dict,
                                       moof_offsets: List[int] = None) -> None:
        if not moof_boxes:
            return
        decode_ends = {}
        decode_starts = {}
        for moof_number, moof_box in enumerate(moof_boxes):
            parsed_moof_box = MediaBoxExtractor.extract_media_boxes(moof_box)
            if not parsed_moof_box:
                MediaDataParser.__logger.error(f'Cannot parse moof box: {moof_box}')
                raise ValueError("Cannot parse moof box")
            moof_atom = MediaBoxExtractor.get_mp4_box(parsed_moof_box, 'moof')
            if not moof_atom:
                MediaDataParser.__logger.error(f'Cannot get moof box from {parsed_moof_box}')
                raise ValueError("There is no 'moof' atom in mp4 data")
            traf_atoms = MediaBoxExtractor.get_all_mp4_sub_boxes(moof_atom, 'traf')
            for traf_number, traf_atom in enumerate(traf_atoms, 1):
                tfhd_atom = MediaBoxExtractor.get_mp4_sub_box(traf_atom, 'tfhd')
                if tfhd_atom is None or tfhd_atom.track_ID not in timescales:
                    raise ValueError('Fragment has a missing or unknown track header')
                track_id = tfhd_atom.track_ID
                timescale = timescales[track_id]
                if timescale <= 0:
                    raise ValueError(f'Fragment requires a positive timescale for track {track_id}')
                tfdt_atom = MediaBoxExtractor.get_mp4_sub_box(traf_atom, 'tfdt')
                decode_time = tfdt_atom.baseMediaDecodeTime if tfdt_atom is not None else decode_ends.get(track_id, 0)
                if moof_timings is not None and track_id in decode_ends and decode_time != decode_ends[track_id]:
                    raise ValueError(f'Unsupported fragment decode gap or overlap for track {track_id}')
                decode_starts.setdefault(track_id, Fraction(decode_time, timescale))
                runs = MediaBoxExtractor.get_all_mp4_sub_boxes(traf_atom, 'trun')
                trun_atom = SimpleNamespace(sample_info=[sample for run in runs for sample in run.sample_info])
                trex_atom = trex_atoms.get(track_id)
                MediaDataParser.__fill_moof_fragment(
                    moof_fragments, tfhd_atom, trun_atom, trex_atom, timescale, moof_timings, decode_time)
                decode_ends[track_id] = decode_time + MediaDataParser.__get_moof_fragment_duration(
                    trun_atom, tfhd_atom, trex_atom)
                if moof_timings is not None:
                    if moof_offsets is not None:
                        moof_timings.setdefault('fragment_locations', {}).setdefault(track_id, []).append(
                            (moof_offsets[moof_number], traf_number))
                    intervals = moof_timings[track_id]
                    if len(intervals) > 1 and intervals[-2][1] != intervals[-1][0]:
                        raise ValueError(f'Unsupported fragment presentation gap or overlap for track {track_id}')
        if moof_timings is not None:
            moof_timings['decode_starts'] = decode_starts

    @staticmethod
    def __set_fragment_index(track: MediaTrackInfo, indexes: dict, fragments: dict,
                             timings: dict, timescale: int) -> None:
        """Validate complete source addressing independently of edited presentation timing."""
        entries = indexes.get(track.track_id, [])
        durations = fragments.get(track.track_id, [[], []])[0]
        locations = timings.get('fragment_locations', {}).get(track.track_id, [])
        if not entries or len(entries) != len(durations) or len(entries) != len(locations):
            raise ValueError(f'Incomplete tfra index for {track.blob_name}, track {track.track_id}')
        chunks = []
        for position, (entry, duration, location) in enumerate(zip(entries, durations, locations)):
            time, offset, traf_number, run_number, sample_number = entry
            if (offset, traf_number) != location or run_number != 1 or sample_number != 1:
                raise ValueError(f'Unsupported tfra fragment reference for {track.blob_name}, track {track.track_id}')
            if position and (time <= entries[position - 1][0] or offset <= entries[position - 1][1]):
                raise ValueError(f'Non-increasing tfra index for {track.blob_name}, track {track.track_id}')
            sample_ticks = duration * timescale
            ticks = entries[position + 1][0] - time if position + 1 < len(entries) else sample_ticks
            if ticks <= 0 or ticks != sample_ticks or sample_ticks.denominator != 1:
                raise ValueError(f'tfra duration does not match fragment samples for {track.blob_name}, track {track.track_id}')
            chunks.append((time, int(ticks)))
        track.fragment_index = chunks
        track.fragment_timescale = timescale

    @staticmethod
    def __update_media_track_info_list(media_data: MediaData) -> None:
        track_names_list = []
        filtered_video_tracks = Common.get_filtered_tracks(media_data.media_track_info_list, TrackType.VIDEO)
        filtered_audio_tracks = Common.get_filtered_tracks(media_data.media_track_info_list, TrackType.AUDIO)
        filtered_text_tracks = Common.get_filtered_tracks(media_data.media_track_info_list, TrackType.TEXT)
        
        # Process audio tracks
        audio_group_counts = {}
        for track in filtered_audio_tracks:
            language_code, language_name = Common.get_language_3_code_and_name(track.language)
            track.language = language_code
            previous_track = track_names_list[-1] if track_names_list else None
            same_timeline = (previous_track is not None and track.is_equal_language(previous_track)
                             and track.is_equal_chunk_data(previous_track))
            if same_timeline and track.is_equal_bitrate(previous_track):
                continue
            if same_timeline and track == previous_track:
                track.track_name = previous_track.track_name
            else:
                index = audio_group_counts.get(language_name, 0)
                track.track_name = language_name + (str(index) if index else '')
                audio_group_counts[language_name] = index + 1
            track_names_list.append(track)
        
        # Process text tracks (CMFT files from VTT conversion)
        text_track_names_list = []
        for track in filtered_text_tracks:
            if track.language and track.language != 'und':
                # Get language code and full name, with fallback to "Undefined"
                try:
                    language_code, language_name = Common.get_language_3_code_and_name(track.language)
                    track.language = language_code
                    track.track_name = f"subs_{language_name}"
                except Exception as e:
                    MediaDataParser.__logger.error(f"Text track - Could not resolve language '{track.language}': {e}")
                    track.track_name = "Undefined"
            else:
                track.track_name = ""
                MediaDataParser.__logger.warning(f"Text track - No language code")
            
            # Handle duplicate language names: count how many same-language tracks
            # already exist and append a digit (1, 2, …) from the second occurrence.
            # This mirrors the audio convention so each track name is unique.
            same_lang_count = sum(
                1 for t in text_track_names_list if t.language == track.language
            )
            if same_lang_count > 0:
                track.track_name = track.track_name + str(same_lang_count)
            text_track_names_list.append(track)
        
        media_data.media_track_info_list = filtered_video_tracks + track_names_list + text_track_names_list
