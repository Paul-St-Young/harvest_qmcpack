module pbcbox
  implicit none
  private

  public :: displacement_table, disp_in_box, pos_in_box
  public :: displacement_ab, displacement12

  contains

  function pos_in_box(x, ax)
    double precision :: x, ax, pos_in_box
    pos_in_box = mod(x, ax)
  end function pos_in_box

  function disp_in_box(dx, ax)
    double precision :: dx, ax, disp_in_box
    disp_in_box = dx - nint(dx/ax)*ax
  end function disp_in_box

  subroutine displacement_table(pos, box, drij, nat, ndim)
    double precision, intent(in) :: pos(nat, ndim), box(ndim)
    double precision, intent(out) :: drij(nat, nat, ndim)
    integer, intent(in) :: nat, ndim
    ! local variables
    integer :: j, iat, jat
    drij(:,:,:) = 0.d0
    do jat=1,nat
    do iat=1,jat-1
    do j=1,ndim
      drij(iat, jat, j) = disp_in_box(pos(iat, j)-pos(jat, j), box(j))
      drij(jat, iat, j) = -drij(iat, jat, j)
    enddo
    enddo
    enddo
  end subroutine displacement_table

  subroutine displacement_ab(posa, posb, box, drij, na, nb, ndim)
    double precision, intent(in) :: posa(na, ndim), posb(nb, ndim), box(ndim)
    double precision, intent(out) :: drij(na, nb, ndim)
    integer, intent(in) :: na, nb, ndim
    ! local variables
    integer :: j, iat, jat
    drij(:,:,:) = 0.d0
    do jat=1,nb
    do iat=1,na
    do j=1,ndim
      drij(iat, jat, j) = disp_in_box(posa(iat, j)-posb(jat, j), box(j))
    enddo
    enddo
    enddo
  end subroutine displacement_ab

  subroutine displacement12(posa, posb, box, drij, na, ndim)
    double precision, intent(in) :: posa(na, ndim), posb(na, ndim), box(ndim)
    double precision, intent(out) :: drij(na, ndim)
    integer, intent(in) :: na, ndim
    ! local variables
    integer :: j, iat
    drij(:,:) = 0.d0
    do iat=1,na
    do j=1,ndim
      drij(iat, j) = disp_in_box(posa(iat, j)-posb(iat, j), box(j))
    enddo
    enddo
  end subroutine displacement12

end module pbcbox
