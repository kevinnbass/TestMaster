import { of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
/**
 * `mapResponse` is a map operator with included error handling.
 * It is similar to `tapResponse`, but allows to map the response as well.
 *
 * The main use case is for NgRx Effects which requires an action to be dispatched.
 *
 * @usageNotes
 * ```ts
 * export const loadAllUsers = createEffect((
 *   actions$ = inject(Actions),
 *   usersService = inject(UsersService)
 * ) => {
 *   return actions$.pipe(
 *     ofType(UsersPageActions.opened),
 *     exhaustMap(() => {
 *       return usersService.getAll().pipe(
 *         mapResponse({
 *           next: (users) => UsersApiActions.usersLoadedSuccess({ users }),
 *           error: (error) => UsersApiActions.usersLoadedFailure({ error }),
 *         })
 *       );
 *     })
 *   );
 * });
 * ```
 */
export function mapResponse(observer) {
    return (source$) => source$.pipe(map((value) => observer.next(value)), catchError((error) => of(observer.error(error))));
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWFwLXJlc3BvbnNlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vbW9kdWxlcy9vcGVyYXRvcnMvc3JjL21hcC1yZXNwb25zZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSxPQUFPLEVBQWMsRUFBRSxFQUFFLE1BQU0sTUFBTSxDQUFDO0FBQ3RDLE9BQU8sRUFBRSxVQUFVLEVBQUUsR0FBRyxFQUFFLE1BQU0sZ0JBQWdCLENBQUM7QUFPakQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0F5Qkc7QUFDSCxNQUFNLFVBQVUsV0FBVyxDQUN6QixRQUEyQztJQUUzQyxPQUFPLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FDakIsT0FBTyxDQUFDLElBQUksQ0FDVixHQUFHLENBQUMsQ0FBQyxLQUFLLEVBQUUsRUFBRSxDQUFDLFFBQVEsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsRUFDcEMsVUFBVSxDQUFDLENBQUMsS0FBSyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQ2pELENBQUM7QUFDTixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgT2JzZXJ2YWJsZSwgb2YgfSBmcm9tICdyeGpzJztcbmltcG9ydCB7IGNhdGNoRXJyb3IsIG1hcCB9IGZyb20gJ3J4anMvb3BlcmF0b3JzJztcblxudHlwZSBNYXBSZXNwb25zZU9ic2VydmVyPFQsIEUsIFIxLCBSMj4gPSB7XG4gIG5leHQ6ICh2YWx1ZTogVCkgPT4gUjE7XG4gIGVycm9yOiAoZXJyb3I6IEUpID0+IFIyO1xufTtcblxuLyoqXG4gKiBgbWFwUmVzcG9uc2VgIGlzIGEgbWFwIG9wZXJhdG9yIHdpdGggaW5jbHVkZWQgZXJyb3IgaGFuZGxpbmcuXG4gKiBJdCBpcyBzaW1pbGFyIHRvIGB0YXBSZXNwb25zZWAsIGJ1dCBhbGxvd3MgdG8gbWFwIHRoZSByZXNwb25zZSBhcyB3ZWxsLlxuICpcbiAqIFRoZSBtYWluIHVzZSBjYXNlIGlzIGZvciBOZ1J4IEVmZmVjdHMgd2hpY2ggcmVxdWlyZXMgYW4gYWN0aW9uIHRvIGJlIGRpc3BhdGNoZWQuXG4gKlxuICogQHVzYWdlTm90ZXNcbiAqIGBgYHRzXG4gKiBleHBvcnQgY29uc3QgbG9hZEFsbFVzZXJzID0gY3JlYXRlRWZmZWN0KChcbiAqICAgYWN0aW9ucyQgPSBpbmplY3QoQWN0aW9ucyksXG4gKiAgIHVzZXJzU2VydmljZSA9IGluamVjdChVc2Vyc1NlcnZpY2UpXG4gKiApID0+IHtcbiAqICAgcmV0dXJuIGFjdGlvbnMkLnBpcGUoXG4gKiAgICAgb2ZUeXBlKFVzZXJzUGFnZUFjdGlvbnMub3BlbmVkKSxcbiAqICAgICBleGhhdXN0TWFwKCgpID0+IHtcbiAqICAgICAgIHJldHVybiB1c2Vyc1NlcnZpY2UuZ2V0QWxsKCkucGlwZShcbiAqICAgICAgICAgbWFwUmVzcG9uc2Uoe1xuICogICAgICAgICAgIG5leHQ6ICh1c2VycykgPT4gVXNlcnNBcGlBY3Rpb25zLnVzZXJzTG9hZGVkU3VjY2Vzcyh7IHVzZXJzIH0pLFxuICogICAgICAgICAgIGVycm9yOiAoZXJyb3IpID0+IFVzZXJzQXBpQWN0aW9ucy51c2Vyc0xvYWRlZEZhaWx1cmUoeyBlcnJvciB9KSxcbiAqICAgICAgICAgfSlcbiAqICAgICAgICk7XG4gKiAgICAgfSlcbiAqICAgKTtcbiAqIH0pO1xuICogYGBgXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBtYXBSZXNwb25zZTxULCBFLCBSMSwgUjI+KFxuICBvYnNlcnZlcjogTWFwUmVzcG9uc2VPYnNlcnZlcjxULCBFLCBSMSwgUjI+XG4pOiAoc291cmNlJDogT2JzZXJ2YWJsZTxUPikgPT4gT2JzZXJ2YWJsZTxSMSB8IFIyPiB7XG4gIHJldHVybiAoc291cmNlJCkgPT5cbiAgICBzb3VyY2UkLnBpcGUoXG4gICAgICBtYXAoKHZhbHVlKSA9PiBvYnNlcnZlci5uZXh0KHZhbHVlKSksXG4gICAgICBjYXRjaEVycm9yKChlcnJvcikgPT4gb2Yob2JzZXJ2ZXIuZXJyb3IoZXJyb3IpKSlcbiAgICApO1xufVxuIl19