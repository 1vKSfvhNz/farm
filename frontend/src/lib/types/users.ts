// lib/types/users.ts

/**
 * Types de base pour les utilisateurs
 * Correspond au schéma Pydantic UserBase du backend
 */
export interface UserBase {
  email: string;
  phone?: string | null;
  username: string;
  full_name: string;
  is_active: boolean;
  roles: string[];
}

// ===== CHAMPS EMPLOYÉ =====

export type EmployeeStatus = 
  | "actif" 
  | "conge" 
  | "maladie" 
  | "suspendu" 
  | "licencie" 
  | "retraite" 
  | "stagiaire";

export type EmployeeType = 
  | "permanent" 
  | "stagiaire" 
  | "contractuel" 
  | "saisonnier" 
  | "consultant";

export interface EmployeeInfo {
  employee_id?: string | null;
  department?: string | null;
  hire_date?: string | null; // ISO date string
  
  // Salaire
  base_salary?: number | null;
  salary_currency?: string;
  salary_frequency?: string;
  bonus?: number;
  
  // Statut
  employee_status?: EmployeeStatus;
  employee_type?: EmployeeType;
  
  // Informations bancaires
  bank_name?: string | null;
  bank_account?: string | null;
  rib?: string | null;
  
  // Informations administratives
  national_id?: string | null;
  social_security_number?: string | null;
  tax_id?: string | null;
  
  // Contacts d'urgence
  emergency_contact_name?: string | null;
  emergency_contact_phone?: string | null;
  
  observations?: string | null;
}

/**
 * Utilisateur complet avec les champs de base de données
 * Correspond au schéma Pydantic UserResponse du backend
 */
export interface UserResponse extends UserBase, EmployeeInfo {
  id: number;
  created_at: string;
  updated_at: string;
}

/**
 * Données pour la création d'un utilisateur
 * Correspond au schéma Pydantic UserCreate du backend
 */
export interface UserCreate extends Omit<UserBase, 'is_active' | 'roles'>, Partial<EmployeeInfo> {
  is_active?: boolean;
  roles?: string[];
}

/**
 * Données pour la mise à jour d'un utilisateur
 * Correspond au schéma Pydantic UserUpdate du backend
*/
export interface UserUpdate extends Partial<Omit<UserBase, 'username'>>, Partial<EmployeeInfo> {
  username?: string;
  password?: string;
}

// lib/types/users.ts (ajout)

export interface UserSearchParams {
    // Pagination
    skip?: number;
    limit?: number;
    
    // Recherche textuelle
    search?: string;
    
    // Filtres de base
    roles?: string[];
    is_active?: boolean;
    
    // Filtres employé
    employee_id?: string;
    department?: string;
    employee_status?: string[];
    employee_type?: string[];
    hire_date_from?: string; // ISO date string
    hire_date_to?: string;   // ISO date string
    
    // Filtres salaire
    salary_min?: number;
    salary_max?: number;
    
    // Filtres dates
    created_from?: string; // ISO datetime
    created_to?: string;   // ISO datetime
    
    // Tri
    order_by?: string;
    order_direction?: 'asc' | 'desc';
}

// ========================================
// Types pour les sessions utilisateur
// ========================================

export interface UserSession {
  id: number;
  user_id: number;
  ip_address: string | null;
  user_agent: string | null;
  device_info: Record<string, unknown> | null;
  created_at: string;
  expires_at: string;
  is_valid: boolean;
  logout_at: string | null;
}

// ========================================
// Types pour les logs d'action
// ========================================

export interface ActionLog {
  id: number;
  user_id: number;
  action: string;
  entity_type: string | null;
  entity_id: number | null;
  details: Record<string, unknown> | null;
  ip_address: string | null;
  created_at: string;
}