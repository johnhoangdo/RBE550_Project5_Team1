(define (domain blocksworld)
  (:requirements :strips :typing)
  (:types block)

  (:predicates
    (on ?x - block ?y - block)       ; block ?x is on top of block ?y
    (ontable ?x - block)             ; block ?x is on the table
    (clear ?x - block)               ; no block is on top of ?x
    (handempty)                      ; gripper is not holding anything
    (holding ?x - block)             ; gripper is holding block ?x
  )

  (:action pick-up
    :parameters (?x - block)
    :precondition (and 
      (clear ?x)           ; nothing on top of block
      (ontable ?x)         ; block is on table
      (handempty)          ; gripper is empty
    )
    :effect (and 
      (holding ?x)         ; now holding the block
      (not (ontable ?x))   ; block no longer on table
      (not (clear ?x))     ; block no longer clear (in gripper)
      (not (handempty))    ; gripper no longer empty
    )
  )

  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)    ; must be holding the block
    :effect (and 
      (ontable ?x)         ; block now on table
      (clear ?x)           ; block is now clear
      (handempty)          ; gripper is now empty
      (not (holding ?x))   ; no longer holding block
    )
  )

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and 
      (holding ?x)         ; must be holding block x
      (clear ?y)           ; block y must be clear
    )
    :effect (and 
      (on ?x ?y)           ; x is now on y
      (clear ?x)           ; x is now clear
      (handempty)          ; gripper is now empty
      (not (holding ?x))   ; no longer holding x
      (not (clear ?y))     ; y is no longer clear
    )
  )

  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and 
      (on ?x ?y)           ; x must be on y
      (clear ?x)           ; x must be clear
      (handempty)          ; gripper must be empty
    )
    :effect (and 
      (holding ?x)         ; now holding x
      (clear ?y)           ; y is now clear
      (not (on ?x ?y))     ; x no longer on y
      (not (clear ?x))     ; x no longer clear
      (not (handempty))    ; gripper no longer empty
    )
  )
)