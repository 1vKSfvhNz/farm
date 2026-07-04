"""modification de users

Revision ID: 3053368a0690
Revises: 5f593e04ee8a
Create Date: 2026-06-26 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ENUM

# revision identifiers, used by Alembic.
revision = '3053368a0690'
down_revision = '5f593e04ee8a'
branch_labels = None
depends_on = None

def upgrade():
    # 1. Créer les types ENUM avant de les utiliser
    employee_status_enum = ENUM(
        'ACTIF', 'CONGE', 'MALADIE', 'SUSPENDU', 
        'LICENCIE', 'RETRAITE', 'STAGIAIRE',
        name='employeestatusenum',
        create_type=True  # Important pour créer le type
    )
    employee_status_enum.create(op.get_bind(), checkfirst=True)
    
    employee_type_enum = ENUM(
        'PERMANENT', 'STAGIAIRE', 'CONTRACTUEL', 
        'SAISONNIER', 'CONSULTANT',
        name='employeetypeenum',
        create_type=True
    )
    employee_type_enum.create(op.get_bind(), checkfirst=True)
    
    # 2. Ajouter les colonnes avec les types ENUM
    op.add_column('users', sa.Column('employee_id', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('position', sa.String(length=100), nullable=True))
    op.add_column('users', sa.Column('department', sa.String(length=100), nullable=True))
    op.add_column('users', sa.Column('hire_date', sa.Date(), nullable=True))
    op.add_column('users', sa.Column('base_salary', sa.Float(), nullable=True))
    op.add_column('users', sa.Column('salary_currency', sa.String(length=3), nullable=True, server_default='XOF'))
    op.add_column('users', sa.Column('salary_frequency', sa.String(length=20), nullable=True, server_default='monthly'))
    op.add_column('users', sa.Column('bonus', sa.Float(), nullable=True, server_default='0.0'))
    
    # Utiliser le type ENUM créé
    op.add_column('users', sa.Column('employee_status', employee_status_enum, nullable=True))
    op.add_column('users', sa.Column('employee_type', employee_type_enum, nullable=True))
    
    op.add_column('users', sa.Column('bank_name', sa.String(length=100), nullable=True))
    op.add_column('users', sa.Column('bank_account', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('rib', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('national_id', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('social_security_number', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('tax_id', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('emergency_contact_name', sa.String(length=100), nullable=True))
    op.add_column('users', sa.Column('emergency_contact_phone', sa.String(length=32), nullable=True))
    op.add_column('users', sa.Column('observations', sa.Text(), nullable=True))
    
    # 3. Créer l'index sur employee_id
    op.create_index('ix_users_employee_id', 'users', ['employee_id'], unique=True)


def downgrade():
    # Supprimer les colonnes dans l'ordre inverse
    op.drop_index('ix_users_employee_id', table_name='users')
    
    op.drop_column('users', 'observations')
    op.drop_column('users', 'emergency_contact_phone')
    op.drop_column('users', 'emergency_contact_name')
    op.drop_column('users', 'tax_id')
    op.drop_column('users', 'social_security_number')
    op.drop_column('users', 'national_id')
    op.drop_column('users', 'rib')
    op.drop_column('users', 'bank_account')
    op.drop_column('users', 'bank_name')
    op.drop_column('users', 'employee_type')
    op.drop_column('users', 'employee_status')
    op.drop_column('users', 'bonus')
    op.drop_column('users', 'salary_frequency')
    op.drop_column('users', 'salary_currency')
    op.drop_column('users', 'base_salary')
    op.drop_column('users', 'hire_date')
    op.drop_column('users', 'department')
    op.drop_column('users', 'employee_id')
    
    # Supprimer les types ENUM
    ENUM(name='employeestatusenum').drop(op.get_bind(), checkfirst=True)
    ENUM(name='employeetypeenum').drop(op.get_bind(), checkfirst=True)