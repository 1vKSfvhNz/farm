// lib/types/pagination.ts
export interface PaginationParams {
    page: number;
    limit: number;
    skip: number;
}

export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    skip: number;
    limit: number;
    page: number;
    total_pages: number;
    has_next: boolean;
    has_prev: boolean;
}