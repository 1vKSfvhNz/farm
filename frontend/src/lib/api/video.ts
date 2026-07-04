// lib/api/video.ts
import { apiClient } from './client';
import type { CameraCreate, CameraUpdate, CameraResponse, VideoRecordResponse } from '../types/video';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const videoApi = {
    getCameras: (params?: Partial<PaginationParams> & { enclos_id?: number; is_active?: boolean }) =>
        apiClient.get<PaginatedResponse<CameraResponse>>('/video/cameras', { params }),

    getCameraStream: (camera_id: number) =>
        apiClient.get<{ stream_url: string }>(`/video/cameras/${camera_id}/stream`),

    getVideoRecords: (params?: Partial<PaginationParams> & { animal_id?: number; enclos_id?: number; action_type?: string; start_date?: string; end_date?: string }) =>
        apiClient.get<PaginatedResponse<VideoRecordResponse>>('/video/records', { params }),

    getVideoRecord: (record_id: number) =>
        apiClient.get<VideoRecordResponse>(`/video/records/${record_id}`),

    getVideosForAnimal: (animal_id: number, limit?: number) =>
        apiClient.get<VideoRecordResponse[]>(`/video/records/animal/${animal_id}`, { params: { limit } }),

    createCamera: (data: CameraCreate) =>
        apiClient.post<CameraResponse>('/video/cameras', data),

    updateCamera: (id: number, data: CameraUpdate) =>
        apiClient.put<CameraResponse>(`/video/cameras/${id}`, data),

    deleteCamera: (id: number) =>
        apiClient.delete(`/video/cameras/${id}`)
};